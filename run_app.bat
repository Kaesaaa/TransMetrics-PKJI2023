@echo off
title TRANS-METRICS: PKJI 2023 & Smart CCTV Monitor
cd /d "%~dp0"
echo ===============================================================================
echo     TRANS-METRICS: Kalkulator Kinerja Ruas Jalan PKJI 2023 ^& Smart CCTV
echo ===============================================================================
echo.
echo [1/2] Menyiapkan environment Python 3.10...
set "PATH=C:\Users\HP\AppData\Local\Programs\Python\Python310\Scripts;C:\Users\HP\AppData\Local\Programs\Python\Python310;%PATH%"

echo [2/2] Menjalankan Streamlit Server di http://localhost:8501...
echo.
echo [INFO] Browser akan terbuka otomatis. 
echo [INFO] Jika tidak terbuka, silakan buka: http://localhost:8501
echo [INFO] Tekan CTRL + C di jendela ini untuk mematikan server.
echo ===============================================================================
echo.

"C:\Users\HP\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py --server.port 8501

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Gagal menjalankan Streamlit secara langsung. Mencoba alternatif...
    streamlit run app.py --server.port 8501
)

pause
