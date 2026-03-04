# Script đơn giản để chạy service (double-click hoặc chạy trực tiếp)

Set-Location $PSScriptRoot

# Kill process cũ nếu có
$port = netstat -ano | findstr :8110
if ($port) {
    $pid = ($port -split '\s+')[-1]
    Write-Host "[INFO] Đang kill process cũ (PID: $pid)..." -ForegroundColor Yellow
    taskkill /F /PID $pid 2>$null
    Start-Sleep -Seconds 1
}

# Activate và chạy
& .\.venv\Scripts\Activate.ps1
python main.py

