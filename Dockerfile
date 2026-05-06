# ============================================================
# Face Recognition Service — Production Dockerfile
# Multi-stage build: slim Python 3.11 + headless OpenCV
# ============================================================

# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

# Cài build dependencies cần cho insightface, opencv, scipy …
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements trước để tận dụng Docker layer cache
COPY requirements.txt .

# Cài dependencies vào thư mục riêng để copy sang runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

# Runtime dependencies (headless OpenCV, libgl cho onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages từ builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source code
COPY main.py .

# Copy trained models (trừ w600k_r50.onnx — file lớn download từ HuggingFace)
COPY trained_models/ ./trained_models/

# Download model recognition từ HuggingFace (166MB, hash đã verify giống máy local)
RUN mkdir -p trained_models/recognition && \
    curl -L -o trained_models/recognition/w600k_r50.onnx \
    "https://huggingface.co/maze/faceX/resolve/e010b5098c3685fd00b22dd2aec6f37320e3d850/w600k_r50.onnx"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_GPU=0

# Expose port
EXPOSE 8110

# Health check — gọi endpoint root mỗi 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8110/ || exit 1

# Chạy service
CMD ["python", "main.py"]
