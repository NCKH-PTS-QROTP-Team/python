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