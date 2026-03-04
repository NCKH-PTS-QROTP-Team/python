@echo off
REM Script đơn giản để chạy service (double-click để chạy)

cd /d "%~dp0"

REM Kill process cũ nếu có
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8110') do (
    echo [INFO] Đang kill process cũ trên port 8110...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 >nul
)

REM Activate và chạy
call .venv\Scripts\activate.bat
python main.py

pause

