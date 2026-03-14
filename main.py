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
