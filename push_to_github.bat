@echo off
title TRANS-METRICS: Push ke GitHub (Kaesaaa/TransMetrics-PKJI2023)
cd /d "%~dp0"

echo ===============================================================================
echo     TRANS-METRICS: UNGGAH REPOSITORI KE GITHUB UNTUK DEPLOYMENT CLOUD
echo ===============================================================================
echo Repositori Target: https://github.com/Kaesaaa/TransMetrics-PKJI2023.git
echo Branch Target    : main
echo.
echo Menyiapkan path Git...
set "PATH=C:\Users\HP\MinGit\cmd;%PATH%"

echo.
echo [OPSI AUTENTIKASI GITHUB]
echo GitHub saat ini mewajibkan Personal Access Token (PAT) untuk push lewat HTTPS.
echo Anda bisa membuat token secara gratis di:
echo https://github.com/settings/tokens/new?scopes=repo^&description=TransMetrics-Token
echo.
set /p TOKEN="Masukkan Personal Access Token GitHub Anda (atau tekan Enter jika sudah login): "

echo.
if "%TOKEN%"=="" (
    echo Mengunggah langsung dengan git push -u origin main...
    git push -u origin main
) else (
    echo Mengunggah menggunakan kredensial token Anda...
    git push -u "https://%TOKEN%@github.com/Kaesaaa/TransMetrics-PKJI2023.git" main
)

echo.
if %errorlevel% equ 0 (
    echo ===============================================================================
    echo [SUKSES 100%%] Seluruh kode berhasil terunggah ke repositori GitHub Anda!
    echo.
    echo Langkah Selanjutnya untuk Deploy ke Streamlit Cloud:
    echo 1. Buka browser ke: https://share.streamlit.io/
    echo 2. Login dengan akun GitHub Anda (Kaesaaa).
    echo 3. Klik tombol "New app".
    echo 4. Pilih Repository: Kaesaaa/TransMetrics-PKJI2023, Branch: main, File: app.py
    echo 5. Klik "Deploy!"
    echo ===============================================================================
) else (
    echo ===============================================================================
    echo [GAGAL] Push belum berhasil. Periksa kembali koneksi internet atau Personal
    echo Access Token Anda (pastikan centang izin 'repo' saat membuat token).
    echo ===============================================================================
)

echo.
pause
