# Script peluncur TRANS-METRICS via PowerShell
Set-Location -Path $PSScriptRoot

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "    TRANS-METRICS: Kalkulator Kinerja Ruas Jalan PKJI 2023 & Smart CCTV" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[1/2] Menyiapkan environment Python 3.10 & Streamlit..." -ForegroundColor Yellow
$env:Path = "C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts;C:\Users\HP\AppData\Local\Programs\Python\Python310;" + $env:Path

Write-Host "[2/2] Menjalankan Server di http://localhost:8501..." -ForegroundColor Green
Write-Host "[INFO] Tekan Ctrl+C di terminal ini untuk menghentikan server." -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

& "C:\Users\HP\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py --server.port 8501
