@echo off
REM Script để chạy Face Recognition Service (Windows Batch)

cd /d "%~dp0"

echo === Face Recognition Service ===

REM Kill process cũ nếu có
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8110') do (
    echo [INFO] Đang kill process cũ trên port 8110...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 >nul
)

REM Kiểm tra virtual environment
if not exist ".venv" (
    echo [INFO] Tạo virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo [INFO] Kích hoạt virtual environment...
call .venv\Scripts\activate.bat

REM Kiểm tra và cài dependencies
echo [INFO] Kiểm tra dependencies...
pip install -q -r requirements.txt

REM Kiểm tra models
echo [INFO] Kiểm tra models...
if not exist "trained_models\detection\det_10g.onnx" (
    echo [ERROR] Không tìm thấy detection model!
    pause
    exit /b 1
)

echo [OK] Tất cả đã sẵn sàng!
echo.
echo Đang khởi động service tại http://localhost:8110
echo Nhấn Ctrl+C để dừng service
echo.

REM Chạy service
python main.py

pause

