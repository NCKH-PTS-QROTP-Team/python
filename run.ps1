# Script để chạy Face Recognition Service (PowerShell)

Write-Host "=== Face Recognition Service ===" -ForegroundColor Cyan

# Kiểm tra virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Tạo virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "[INFO] Kích hoạt virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Kiểm tra và cài dependencies
Write-Host "[INFO] Kiểm tra dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt

# Kiểm tra models
Write-Host "[INFO] Kiểm tra models..." -ForegroundColor Yellow
if (-not (Test-Path "trained_models\detection\det_10g.onnx")) {
    Write-Host "[ERROR] Không tìm thấy detection model!" -ForegroundColor Red
    Read-Host "Nhấn Enter để thoát"
    exit 1
}

Write-Host "[OK] Tất cả đã sẵn sàng!" -ForegroundColor Green
Write-Host ""
Write-Host "Đang khởi động service tại http://localhost:8110" -ForegroundColor Cyan
Write-Host "Nhấn Ctrl+C để dừng service" -ForegroundColor Yellow
Write-Host ""

# Chạy service
python main.py

