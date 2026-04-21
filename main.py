"""
Face Recognition Service - Sử dụng models local (copy từ checkin_face_anti_spoofing)
Service riêng để detect mặt và verify mặt, tích hợp với Spring Boot backend.
"""
from pathlib import Path
from typing import Optional, List
import base64
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Lock
import warnings
import re
from collections import deque
from PIL import Image
import io
import signal
from contextlib import contextmanager

# Suppress warnings - giống app.py (line 220)
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')
warnings.filterwarnings('ignore', category=UserWarning)
# Set ONNX Runtime logger severity để chỉ show errors (giống app.py line 242)
import onnxruntime as ort
ort.set_default_logger_severity(3)  # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal

# Import từ insightface (cần cài: pip install insightface)
# Nếu không có insightface, sẽ dùng OpenCV để detect (nhưng không có extract-encoding/verify)
try:
    from insightface.model_zoo import get_model
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("[WARNING] insightface not installed. Install with: pip install insightface")
    print("[WARNING] Service will use OpenCV for detection (no extract-encoding/verify)")
    get_model = None
    face_align = None
    INSIGHTFACE_AVAILABLE = False

app = FastAPI(title="Face Recognition Service", version="1.0.0")

# CORS: cho phép frontend (web) gọi thẳng từ tab face-demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ========== CONFIGURATION ==========
BASE_DIR = Path(__file__).resolve().parent
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

# Model paths (copy từ repo checkin_face_anti_spoofing về thư mục trained_models/)
# Yêu cầu tối thiểu:
#   - trained_models/detection/det_10g.onnx
#   - trained_models/face_anti_spoofing/weights/antispoof_80x80.onnx
#   - (tuỳ chọn, để verify bằng ArcFace) trained_models/recognition/w600k_r50.onnx
DET_MODEL_PATH = TRAINED_MODELS_DIR / "detection" / "det_10g.onnx"
REC_MODEL_PATH = TRAINED_MODELS_DIR / "recognition" / "w600k_r50.onnx"
ANTI_SPOOF_MODEL_PATH = (
    TRAINED_MODELS_DIR
    / "face_anti_spoofing"
    / "weights"
    / "antispoof_80x80.onnx"
)

# GPU Configuration
USE_GPU = True  # Có thể set qua env: os.environ.get('USE_GPU', '1') == '1'
if USE_GPU:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ctx_id = 0
else:
    providers = ['CPUExecutionProvider']
    ctx_id = -1

# Detection & Recognition thresholds
DET_THRESH = 0.6
NMS_THRESH = 0.5
VERIFY_THRESHOLD = 0.7  # Cosine similarity threshold để verify
ANTI_SPOOF_THRESHOLD = 0.55  # Liveness threshold

# Anti-spoof quality gates (từ app.py - logic tốt hơn)
MIN_FACE_SIZE = 120  # Mặt nhỏ hơn cạnh ngắn này coi là kém chất lượng
BLUR_VAR_THR = 250.0  # var Laplacian < BLUR_VAR_THR coi là mờ
MOTION_THR = 2.0  # Ngưỡng chuyển động (tune theo camera)
WINDOW_N = 5  # 5-frame window cho temporal gate

# Thread locks
detection_lock = Lock()
recognition_lock = Lock()
anti_spoof_lock = Lock()

# ========== LOAD MODELS ==========
print(f"[INFO] Loading models from: {TRAINED_MODELS_DIR}")
print(f"[INFO] GPU Mode: {USE_GPU}")

# Load detection model (SCRFD) hoặc OpenCV fallback
scrfd = None
opencv_cascade = None

if INSIGHTFACE_AVAILABLE and get_model is not None:
    try:
        scrfd = get_model(str(DET_MODEL_PATH), providers=providers)
        scrfd.prepare(ctx_id=ctx_id, input_size=(480, 480), det_thresh=DET_THRESH, nms=NMS_THRESH)
        print(f"[OK] SCRFD Detection model loaded: {DET_MODEL_PATH.name}")
    except Exception as e:
        print(f"[WARNING] Failed to load SCRFD model: {e}")
        print("[INFO] Will use OpenCV Haar Cascade for detection")
        scrfd = None

if scrfd is None:
    # Fallback: Dùng OpenCV Haar Cascade (built-in, không cần download)
    opencv_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("[OK] Using OpenCV Haar Cascade for face detection (fallback)")

# Load recognition model (ArcFace) - OPTIONAL
rec_model = None
try:
    if REC_MODEL_PATH.exists():
        available_providers = ort.get_available_providers()
        providers_used = [p for p in providers if p in available_providers] or ['CPUExecutionProvider']
        rec_session = ort.InferenceSession(str(REC_MODEL_PATH), providers=providers_used)
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        rec_model = ArcFaceONNX(str(REC_MODEL_PATH), session=rec_session)
        rec_model.prepare(ctx_id=ctx_id, input_size=(112, 112))
        print(f"[OK] Recognition model loaded: {REC_MODEL_PATH.name}")
    else:
        print(f"[INFO] Recognition model not found at {REC_MODEL_PATH}, embedding/verify endpoints will be disabled.")
except Exception as e:
    print(f"[WARNING] Failed to load recognition model (optional): {e}")
    rec_model = None

# Load anti-spoof model
try:
    anti_providers = [p for p in providers if p in ort.get_available_providers()] or ['CPUExecutionProvider']
    anti_sess = ort.InferenceSession(str(ANTI_SPOOF_MODEL_PATH), providers=anti_providers)
    print(f"[OK] Anti-spoof model loaded: {ANTI_SPOOF_MODEL_PATH.name}")
except Exception as e:
    print(f"[WARNING] Anti-spoof model not available: {e}")
    anti_sess = None

print("[OK] Face Recognition Service ready!")

# ========== UTILITY FUNCTIONS ==========
def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize embedding vector"""
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to BGR image"""
    if not base64_str:
        raise ValueError("Base64 string is empty")
    
    # Remove data URL prefix nếu có
    # Có thể có format: data:image/png;base64,xxx hoặc dataimage/pngbase64xxx (thiếu dấu : và ;)
    import re
    
    # Tìm vị trí "base64" (case-insensitive) và lấy TẤT CẢ phần sau nó
    base64_pos = base64_str.lower().find('base64')
    if base64_pos >= 0:
        # Lấy phần sau "base64" (bỏ qua dấu phẩy và whitespace nếu có)
        base64_str = base64_str[base64_pos + 6:]  # "base64" có 6 ký tự
        # Loại bỏ dấu phẩy và whitespace ở đầu
        base64_str = base64_str.lstrip(', \t\n\r')
        # QUAN TRỌNG: Loại bỏ mọi ký tự không phải base64 ngay sau khi extract
        # (có thể còn sót ký tự từ prefix như "png", "jpg", "dataimage", v.v.)
        # Tìm pattern base64 hợp lệ: PNG bắt đầu bằng "iVBORw0KGgo", JPEG bắt đầu bằng "/9j/"
        # Hoặc tìm vị trí đầu tiên có ký tự base64 hợp lệ (A-Z, a-z, 0-9, +, /, =)
        # nhưng không phải là từ khóa như "dataimage", "png", "jpg"
        
        # Thử tìm pattern PNG hoặc JPEG trước (đáng tin cậy nhất)
        png_pattern = base64_str.find('iVBORw0KGgo')
        jpeg_pattern = base64_str.find('/9j/')
        
        if png_pattern >= 0:
            base64_str = base64_str[png_pattern:]
            print(f"[DEBUG] Found PNG pattern, removed {png_pattern} invalid characters from start")
        elif jpeg_pattern >= 0:
            base64_str = base64_str[jpeg_pattern:]
            print(f"[DEBUG] Found JPEG pattern, removed {jpeg_pattern} invalid characters from start")
        else:
            # Không tìm thấy pattern → tìm vị trí đầu tiên có ký tự base64 hợp lệ
            # nhưng bỏ qua các từ khóa như "dataimage", "png", "jpg"
            base64_start = 0
            for i in range(len(base64_str)):
                char = base64_str[i]
                if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=':
                    # Kiểm tra xem có phải là từ khóa không phải base64 không
                    remaining = base64_str[i:].lower()
                    if remaining.startswith('dataimage') or remaining.startswith('png') or remaining.startswith('jpg'):
                        continue
                    # Nếu ký tự này là base64 hợp lệ và không phải từ khóa → đây là vị trí bắt đầu
                    base64_start = i
                    break
            if base64_start > 0:
                base64_str = base64_str[base64_start:]
                print(f"[DEBUG] Removed {base64_start} invalid characters from start")
        print(f"[DEBUG] Found base64 prefix, extracted base64 data (length: {len(base64_str)})")
    else:
        # Nếu không tìm thấy "base64", thử remove "data:" hoặc "dataimage" prefix
        if base64_str.startswith("data:") or base64_str.startswith("dataimage"):
            # Tìm dấu phẩy cuối cùng (nếu có)
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            else:
                # Nếu không có dấu phẩy, tìm pattern image/xxx và lấy phần sau
                img_match = re.search(r'image/[^/]+([A-Za-z0-9+/=]+)', base64_str, re.IGNORECASE)
                if img_match:
                    base64_str = img_match.group(1)
                else:
                    # Fallback: loại bỏ "dataimage" hoặc "data:" prefix
                    base64_str = re.sub(r'^data:?image/[^/]+', '', base64_str, flags=re.IGNORECASE)
    
    # Loại bỏ whitespace (spaces, newlines, tabs)
    base64_str = base64_str.strip().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    
    # Chỉ giữ lại các ký tự base64 hợp lệ (A-Z, a-z, 0-9, +, /, =)
    # Loại bỏ mọi ký tự không hợp lệ
    # QUAN TRỌNG: Làm điều này TRƯỚC khi thêm padding
    original_length = len(base64_str)
    base64_str = re.sub(r'[^A-Za-z0-9+/=]', '', base64_str)
    if len(base64_str) != original_length:
        print(f"[DEBUG] Removed {original_length - len(base64_str)} invalid characters from base64 string")
    
    if not base64_str:
        raise ValueError("Base64 string is empty after cleaning")
    
    # Thêm padding nếu thiếu (base64 cần padding để decode đúng)
    # Base64 padding là dấu '=' ở cuối, số lượng padding = (4 - len % 4) % 4
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    
    # Log base64 preview để debug
    preview = base64_str[:100] + "..." if len(base64_str) > 100 else base64_str
    print(f"[DEBUG] Base64 string preview: {preview}, length: {len(base64_str)}")
    
    try:
        # Thử decode với validate=True trước (strict)
        img_bytes = base64.b64decode(base64_str, validate=True)
    except Exception as e:
        # Nếu fail, thử decode không validate (lenient) - giống app.py
        try:
            print(f"[WARNING] Base64 decode with validate=True failed: {e}, trying validate=False...")
            img_bytes = base64.b64decode(base64_str, validate=False)
            print("[OK] Base64 decode with validate=False succeeded")
        except Exception as e2:
            # Có thể base64 bị encode 2 lần - thử decode 2 lần
            try:
                print(f"[WARNING] Base64 decode failed, trying double decode (maybe double-encoded)...")
                # Decode lần 1 → bytes
                first_decode_bytes = base64.b64decode(base64_str, validate=False)
                # Decode lần 2 → nếu lần 1 ra bytes hợp lệ, thử decode tiếp
                # Nếu lần 1 ra string (base64), decode tiếp
                try:
                    # Thử decode như bytes (nếu là base64 string đã encode 2 lần)
                    first_decode_str = first_decode_bytes.decode('utf-8', errors='ignore')
                    # Nếu decode được thành string → có thể là base64 string
                    if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in first_decode_str[:100]):
                        print(f"[DEBUG] First decode looks like base64 string, decoding again...")
                        img_bytes = base64.b64decode(first_decode_str, validate=False)
                        print("[OK] Double decode succeeded - base64 was double-encoded!")
                    else:
                        # Không phải base64 string → dùng bytes từ lần 1
                        img_bytes = first_decode_bytes
                        print("[OK] Using first decode result")
                except:
                    # Nếu không decode được string → dùng bytes từ lần 1
                    img_bytes = first_decode_bytes
                    print("[OK] Using first decode result (not double-encoded)")
            except Exception as e3:
                # Log một phần base64 để debug (không log toàn bộ vì quá dài)
                print(f"[ERROR] Failed to decode base64 with all methods. Preview: {preview}, length: {len(base64_str)}")
                print(f"[ERROR] validate=True error: {e}")
                print(f"[ERROR] validate=False error: {e2}")
                print(f"[ERROR] double-decode error: {e3}")
                raise ValueError(f"Invalid base64 image: {e3}") from e3
    
    if len(img_bytes) == 0:
        raise ValueError("Decoded image bytes is empty")
    
    # Log thông tin debug
    print(f"[DEBUG] Decoded image bytes length: {len(img_bytes)}")
    print(f"[DEBUG] First 20 bytes (hex): {img_bytes[:20].hex()}")
    
    # Kiểm tra magic bytes để xác định format
    if img_bytes[:4] == b'\xff\xd8\xff\xe0' or img_bytes[:2] == b'\xff\xd8':
        print("[DEBUG] Detected JPEG format")
    elif img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        print("[DEBUG] Detected PNG format")
    elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
        print("[DEBUG] Detected GIF format")
    elif img_bytes[:2] == b'BM':
        print("[DEBUG] Detected BMP format")
    else:
        print(f"[WARNING] Unknown image format. Magic bytes: {img_bytes[:8].hex()}")
    
    # Dùng PIL làm chính (logic từ app.py - robust hơn OpenCV)
    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        print(f"[DEBUG] PIL opened successfully. Format: {pil_image.format}, Mode: {pil_image.mode}, Size: {pil_image.size}")
        
        # Convert to RGB nếu cần (PIL hỗ trợ nhiều format hơn)
        if pil_image.mode in ('RGBA', 'P', 'LA'):
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array (RGB)
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print(f"[DEBUG] Successfully converted to BGR. Shape: {img.shape}")
        
    except Exception as pil_error:
        print(f"[ERROR] PIL decode failed: {type(pil_error).__name__}: {pil_error}")
        # Fallback: thử OpenCV nếu PIL fail
        try:
            print("[WARNING] PIL decode failed, trying OpenCV fallback...")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV imdecode returned None")
            print(f"[OK] OpenCV fallback successful. Shape: {img.shape}")
        except Exception as cv_error:
            print(f"[ERROR] OpenCV fallback also failed: {type(cv_error).__name__}: {cv_error}")
            # Log thêm thông tin để debug
            print(f"[ERROR] Image bytes length: {len(img_bytes)}")
            print(f"[ERROR] First 50 bytes: {img_bytes[:50]}")
            raise ValueError(f"Cannot decode image from base64 bytes. PIL: {pil_error}, OpenCV: {cv_error}")
    
    if img is None or img.size == 0:
        raise ValueError("Cannot decode image from base64 bytes (all methods failed)")
    
    return img

def encode_image_to_base64(img: np.ndarray) -> str:
    """Encode BGR image to base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')

def detect_faces(img: np.ndarray, max_num: int = 1):
    """Detect faces using SCRFD model (if available) or OpenCV Haar Cascade
    Logic giống app.py line 424-427: gọi trực tiếp scrfd.detect() với lock
    """
    if scrfd is not None:
        # Dùng SCRFD - giống app.py (line 424-427)
        with detection_lock:
            bboxes, kpss = scrfd.detect(img, max_num=max_num)
        return bboxes, kpss
    else:
        # Fallback: Dùng OpenCV Haar Cascade
        if opencv_cascade is None:
            raise RuntimeError("No face detector available. Install insightface or ensure OpenCV is available.")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        faces = opencv_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return np.array([]), None
        
        # Convert to SCRFD format: [x1, y1, x2, y2, conf]
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append([x, y, x+w, y+h, 0.9])  # conf = 0.9
        
        bboxes = np.array(bboxes[:max_num])
        return bboxes, None  # Haar Cascade không có landmarks

def embed_aligned112(img112_bgr: np.ndarray) -> np.ndarray:
    """
    Extract embedding từ aligned 112x112 face image (logic từ app.py)
    """
    if rec_model is None:
        raise RuntimeError("Recognition model (ArcFace) is not loaded")
    
    # Đảm bảo là BGR và đúng kích thước
    if img112_bgr.ndim == 2:
        img112_bgr = cv2.cvtColor(img112_bgr, cv2.COLOR_GRAY2BGR)
    if img112_bgr.shape[:2] != (112, 112):
        img112_bgr = cv2.resize(img112_bgr, (112, 112), interpolation=cv2.INTER_AREA)
    
    with recognition_lock:
        feat = rec_model.get_feat(img112_bgr)
    return l2_normalize(feat)

def extract_face_embedding(img: np.ndarray, kps: np.ndarray, use_enhancement: bool = False) -> np.ndarray:
    """
    Extract face embedding using ArcFace model (logic từ app.py).
    
    Args:
        img: BGR image
        kps: Face landmarks (5 points)
        use_enhancement: Nếu True, sẽ enhance face trước khi extract (tốt hơn)
    
    Returns:
        Normalized embedding vector
    """
    if rec_model is None:
        raise RuntimeError("Recognition model (ArcFace) is not loaded")
    
    # Align face về 112x112 (logic từ app.py)
    aligned = face_align.norm_crop(img, landmark=kps, image_size=112)
    
    # Enhance face nếu cần (logic từ app.py - tốt hơn)
    if use_enhancement:
        aligned, _ = enhance_face_auto(aligned)
    
    # Extract embedding
    return embed_aligned112(aligned)

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

# ========== FACE ENHANCEMENT (từ app.py) ==========
def enhance_face_auto(
    face_bgr,
    denoise_strength=8,         # 5–10: khử noise nhẹ
    clahe_clip=2.0, clahe_grid=8,
    usm_sigma=1.2,              # radius sharpen
    amount_min=0.4, amount_max=1.8,
    low_thr=15.0, high_thr=80.0,
    gamma_corr=True
):
    """
    Smart enhancer:
    1️⃣  Bilateral denoise giữ chi tiết
    2️⃣  Auto exposure (gamma correction)
    3️⃣  CLAHE + Unsharp Mask (adaptive amount)
    """
    # --- Step 1: Denoise nhẹ (Bilateral) ---
    img_dn = cv2.bilateralFilter(face_bgr, d=0,
                                 sigmaColor=denoise_strength,
                                 sigmaSpace=denoise_strength)

    # --- Step 2: Auto exposure (Gamma correction) ---
    if gamma_corr:
        ycc = cv2.cvtColor(img_dn, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]
        meanY = np.mean(y)
        gamma = np.interp(meanY, [50, 180], [1.4, 0.7])  # tối -> tăng sáng
        gamma = np.clip(gamma, 0.8, 1.4)
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        img_gamma = cv2.LUT(img_dn, table)
    else:
        img_gamma = img_dn

    # --- Step 3: CLAHE + Unsharp Mask ---
    ycc = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]
    lapv0 = cv2.Laplacian(y, cv2.CV_64F).var()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                            tileGridSize=(clahe_grid, clahe_grid))
    y_eq = clahe.apply(y)

    amount = np.interp(lapv0, [low_thr, high_thr],
                       [amount_max, amount_min])
    amount = float(np.clip(amount, amount_min, amount_max))

    blur = cv2.GaussianBlur(y_eq, (0, 0), usm_sigma)
    detail = cv2.subtract(y_eq, blur)
    y_sharp = cv2.addWeighted(y_eq, 1.0, detail, amount, 0)

    ycc[:, :, 0] = np.clip(y_sharp, 0, 255).astype(np.uint8)
    sharp_bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    # --- Thống kê debug ---
    y2 = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    lapv1 = cv2.Laplacian(y2, cv2.CV_64F).var()
    meanY2 = np.mean(y2)
    meta = {
        "lapv_before": float(lapv0),
        "lapv_after": float(lapv1),
        "amount": amount,
        "gamma": gamma if gamma_corr else 1.0,
        "meanY": float(meanY),
        "meanY_after": float(meanY2)
    }
    return sharp_bgr, meta

def improved_lap_var(face_bgr):
    """Improved blur detection using Y channel with Gaussian blur"""
    y = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    y = cv2.GaussianBlur(y, (3,3), 0)
    lapv = cv2.Laplacian(y, cv2.CV_64F).var()
    return lapv

def preprocess_anti_spoof(face_bgr: np.ndarray, size=(80, 80)) -> np.ndarray:
    """Preprocess face crop for anti-spoof model"""
    img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
    x = img_rgb.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.ascontiguousarray(x)[None, ...]  # NCHW
    return x

def check_liveness(face_bgr: np.ndarray, use_enhancement: bool = True) -> float:
    """
    Check liveness using anti-spoof model. Returns real probability [0,1]
    Nếu use_enhancement=True, sẽ enhance ảnh trước khi check (tốt hơn)
    """
    if anti_sess is None:
        return 1.0  # If model not loaded, assume real
    
    # Enhance face trước khi check (logic từ app.py - tốt hơn)
    if use_enhancement:
        face_enhanced, _ = enhance_face_auto(face_bgr)
    else:
        face_enhanced = face_bgr
    
    x = preprocess_anti_spoof(face_enhanced, (80, 80))
    with anti_spoof_lock:
        logits = anti_sess.run(["logits"], {"input": x})[0]  # [1, C]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = (e / e.sum(axis=1, keepdims=True))[0]  # (C,)
    # Index 1 = real, 0 = print, 2 = replay
    real_prob = float(probs[1])
    return real_prob

# ========== REQUEST/RESPONSE MODELS ==========
class DetectRequest(BaseModel):
    base64Image: str

class DetectResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class ExtractEncodingRequest(BaseModel):
    base64Image: str

class ExtractEncodingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class VerifyRequest(BaseModel):
    base64Image: str
    registeredEncoding: List[float]  # List of floats from Java

class VerifyResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class AntiSpoofRequest(BaseModel):
    base64Image: str

class AntiSpoofResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

# ========== API ENDPOINTS ==========
@app.get("/")
def root():
    return {
        "service": "Face Recognition Service",
        "version": "1.0.0",
        "status": "running",
        "models": {
            "detection": "SCRFD (det_10g.onnx)",
            "recognition": "ArcFace (w600k_r50.onnx)",
            "anti_spoof": "MiniFASNet (antispoof_80x80.onnx)" if anti_sess else None
        }
    }

@app.post("/api/detect", response_model=DetectResponse)
def detect_faces_endpoint(request: DetectRequest):
    """Detect faces in image. Returns bounding boxes and landmarks."""
    try:
        img = decode_base64_image(request.base64Image)
        bboxes, kpss = detect_faces(img, max_num=1)
        
        if len(bboxes) == 0:
            return DetectResponse(
                success=False,
                message="Không tìm thấy khuôn mặt trong ảnh"
            )
        
        # Get best face (highest confidence)
        bbox = bboxes[0]
        x1, y1, x2, y2, conf = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(bbox[4])
        
        # Draw bounding box on image
        img_annotated = img.copy()
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        processed_image_base64 = encode_image_to_base64(img_annotated)
        
        landmarks = None
        if kpss is not None and len(kpss) > 0:
            landmarks = kpss[0].tolist()
        
        h, w = img.shape[:2]
        return DetectResponse(
            success=True,
            message="Phát hiện khuôn mặt thành công",
            data={
                "faceCount": 1,
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                },
                "confidence": conf,
                "landmarks": landmarks,
                "processedImageBase64": processed_image_base64,
                "imageWidth": w,
                "imageHeight": h
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi detect face: {str(e)}")

@app.post("/api/extract-encoding", response_model=ExtractEncodingResponse)
def extract_encoding_endpoint(request: ExtractEncodingRequest):
    """Extract face encoding from image. Used for registration."""
    try:
        print(f"[DEBUG] Extract encoding request received, base64 length: {len(request.base64Image) if request.base64Image else 0}")
        
        if not INSIGHTFACE_AVAILABLE or rec_model is None:
            print("[ERROR] Recognition model not available")
            return ExtractEncodingResponse(
                success=False,
                message="Recognition model (ArcFace) không khả dụng. Cần cài insightface để sử dụng tính năng này."
            )
        
        print("[DEBUG] Decoding base64 image...")
        img = decode_base64_image(request.base64Image)
        print(f"[DEBUG] Image decoded: shape={img.shape}")
        
        print("[DEBUG] Detecting faces...")
        print(f"[DEBUG] Image shape before detection: {img.shape}")
        try:
            bboxes, kpss = detect_faces(img, max_num=1)
            print(f"[DEBUG] Detected {len(bboxes)} face(s), landmarks: {len(kpss) if kpss is not None else 0}")
        except Exception as detect_error:
            print(f"[ERROR] Face detection failed: {type(detect_error).__name__}: {detect_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Lỗi detect face: {str(detect_error)}")
        
        if len(bboxes) == 0:
            return ExtractEncodingResponse(
                success=False,
                message="Không tìm thấy khuôn mặt trong ảnh"
            )
        
        if kpss is None or len(kpss) == 0:
            return ExtractEncodingResponse(
                success=False,
                message="Không tìm thấy landmarks khuôn mặt"
            )
        
        print("[DEBUG] Extracting face embedding...")
        # Extract embedding với enhancement (logic từ app.py - tốt hơn)
        kps = kpss[0]
        embedding = extract_face_embedding(img, kps, use_enhancement=True)
        print(f"[DEBUG] Embedding extracted: shape={embedding.shape}")
        
        # Flatten embedding nếu là nested array (shape (1, 512) -> (512,))
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        print(f"[DEBUG] Embedding after flatten: shape={embedding.shape}")
        
        # Crop face for visualization
        bbox = bboxes[0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        face_crop = img[y1:y2, x1:x2]
        processed_image_base64 = encode_image_to_base64(face_crop)
        
        print("[DEBUG] Extract encoding successful")
        return ExtractEncodingResponse(
            success=True,
            message="Trích xuất face encoding thành công",
            data={
                "faceEncoding": embedding.tolist(),  # Convert to flat list for JSON
                "processedImageBase64": processed_image_base64,
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            }
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Extract encoding failed: {str(e)}")
        print(f"[ERROR] Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Lỗi extract encoding: {str(e)}")

@app.post("/api/verify", response_model=VerifyResponse)
def verify_face_endpoint(request: VerifyRequest):
    """Verify face by comparing with registered encoding."""
    try:
        if not INSIGHTFACE_AVAILABLE or rec_model is None:
            return VerifyResponse(
                success=False,
                message="Recognition model (ArcFace) không khả dụng. Cần cài insightface để sử dụng tính năng này."
            )
        img = decode_base64_image(request.base64Image)
        bboxes, kpss = detect_faces(img, max_num=1)
        
        if len(bboxes) == 0:
            return VerifyResponse(
                success=False,
                message="Không tìm thấy khuôn mặt trong ảnh"
            )
        
        if kpss is None or len(kpss) == 0:
            return VerifyResponse(
                success=False,
                message="Không tìm thấy landmarks khuôn mặt"
            )
        
        # Crop face với pad (logic từ app.py)
        bbox = bboxes[0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        pad = 50
        h, w = img.shape[:2]
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)
        y2p = min(h, y2 + pad)
        face_crop = img[y1p:y2p, x1p:x2p]
        
        # Quality gate: Kiểm tra kích thước và độ nét
        min_side = min(face_crop.shape[:2]) if face_crop.size else 0
        if min_side < MIN_FACE_SIZE:
            return VerifyResponse(
                success=False,
                message=f"Khuôn mặt quá nhỏ ({min_side}px < {MIN_FACE_SIZE}px)",
                data={
                    "isMatch": False,
                    "similarity": 0.0,
                    "threshold": VERIFY_THRESHOLD,
                    "message": f"Khuôn mặt quá nhỏ ({min_side}px < {MIN_FACE_SIZE}px)"
                }
            )
        
        lapv = improved_lap_var(face_crop)
        if lapv < BLUR_VAR_THR:
            return VerifyResponse(
                success=False,
                message=f"Ảnh quá mờ (LapVar {lapv:.2f} < {BLUR_VAR_THR})",
                data={
                    "isMatch": False,
                    "similarity": 0.0,
                    "threshold": VERIFY_THRESHOLD,
                    "message": f"Ảnh quá mờ (LapVar {lapv:.2f} < {BLUR_VAR_THR})"
                }
            )
        
        # Extract embedding với enhancement (logic từ app.py - tốt hơn)
        kps = kpss[0]
        current_embedding = extract_face_embedding(img, kps, use_enhancement=True)
        
        # Normalize registered encoding trước khi so sánh
        registered_emb = np.array(request.registeredEncoding, dtype=np.float32)
        registered_emb = l2_normalize(registered_emb)  # Đảm bảo normalized
        
        # Compare với cosine similarity (logic từ app.py)
        similarity = cosine_similarity(current_embedding, registered_emb)
        
        is_match = similarity >= VERIFY_THRESHOLD
        
        return VerifyResponse(
            success=True,
            message="Xác thực thành công" if is_match else "Face không khớp",
            data={
                "isMatch": is_match,
                "similarity": similarity,
                "threshold": VERIFY_THRESHOLD,
                "message": (
                    f"Xác thực thành công! Độ tương đồng: {similarity*100:.2f}%"
                    if is_match
                    else f"Face không khớp. Độ tương đồng: {similarity*100:.2f}% (yêu cầu: {VERIFY_THRESHOLD*100:.0f}%)"
                )
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi verify face: {str(e)}")
