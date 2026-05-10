# 🚀 Quick Start - Chạy Service

## Cách 1: Double-click (Đơn giản nhất) ⭐
**Chỉ cần double-click vào file:**
- `start.bat` (Windows)
- `start.ps1` (PowerShell)

Script sẽ tự động:
- Kill process cũ trên port 8110 (nếu có)
- Activate virtual environment
- Chạy service

## Cách 2: Dùng script đầy đủ
**Chạy script có kiểm tra dependencies:**
```powershell
.\run.bat
# hoặc
.\run.ps1
```

## Cách 3: Chạy thủ công
```powershell
cd AI/face-recognition-service
.\.venv\Scripts\Activate.ps1
python main.py
```

## Dừng Service
- Nhấn `Ctrl+C` trong terminal
- Hoặc kill port: `netstat -ano | findstr :8110` rồi `taskkill /F /PID <PID>`

## Service URL
- **API:** http://localhost:8110
- **Docs:** http://localhost:8110/docs

