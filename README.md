# 🚦 TRANS-METRICS: Kalkulator Kinerja Ruas Jalan PKJI 2023 & Smart CCTV Monitor YOLO11

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/YOLO-11%20Nano%2FSmall-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.8%2B-green.svg)](https://opencv.org/)
[![ByteTrack MOT](https://img.shields.io/badge/MOT-ByteTrack%20v2.1-red.svg)](https://github.com/ifzhang/ByteTrack)
[![Pedoman PKJI 2023](https://img.shields.io/badge/Standar-PKJI%202023%20Bina%20Marga-teal.svg)](https://binamarga.pu.go.id/)

> **Aplikasi Web Interaktif Rekayasa Lalu Lintas & Pemantauan Arus Cerdas Dishub**  
> *Dikembangkan untuk Keperluan Akademik Rekayasa Transportasi & Aplikasi Komputer — Program Studi S1 Teknik Sipil.*

---

## 📌 Ringkasan Proyek

**TRANS-METRICS** adalah sistem rekayasa transportasi terintegrasi yang menjembatani streaming video CCTV mentah di lapangan menjadi metrik evaluasi kinerja jalan berstandar nasional **Pedoman Kapasitas Jalan Indonesia (PKJI 2023)** dari Direktorat Jenderal Bina Marga Kementerian PUPR.

Aplikasi ini memiliki 2 Tab Utama:
1. **TAB 1: Kalkulator Kinerja Ruas Jalan (PKJI 2023)**: Evaluasi analitis Derajat Kejenuhan ($DS$) dan Tingkat Pelayanan (*Level of Service* / $LOS$) dengan interpolasi presisi geometrik jalan.
2. **TAB 2: Integrasi Smart CCTV Dishub (Real-Time AI Tracking)**: Pemrosesan video CCTV lalu lintas secara otomatis menggunakan **OpenCV + Ultralytics YOLO11 + ByteTrack MOT**, dilengkapi **Universal CCTV Resolver** (YouTube Live, Web CCTV Pemkot, HLS `.m3u8`, RTSP, Webcam) serta peningkatan kontras malam hari (**CLAHE Auto-Enhance**).

---

## 🌟 Fitur Unggulan

### 📊 TAB 1: Kalkulator Kinerja Ruas Jalan (PKJI 2023)
* **Sidebar Geometrik & Lingkungan Jalan:**
  * **Tipe Jalan Perkotaan:**
    * `2/2 TT` (Dua Lajur Dua Arah Tak-Terbagi) $\rightarrow C_0 = 2900 \text{ smp/jam}$
    * `4/2 T` (Empat Lajur Dua Arah Terbagi) $\rightarrow C_0 = 3300 \text{ smp/jam/arah}$
    * `4/2 TT` (Empat Lajur Dua Arah Tak-Terbagi) $\rightarrow C_0 = 3000 \text{ smp/jam/arah}$
  * **Lebar Jalur Lalu Lintas ($W_L$ / $W_e$):** Slider $5.0 - 14.0\text{ meter}$ dengan interpolasi linier presisi tabel PKJI 2023 untuk faktor $FC_{LJ}$.
  * **Lebar Bahu Efektif ($W_S$):** Slider $0.5 - 3.0\text{ meter}$ untuk faktor penyesuaian hambatan samping ($FC_{HS}$).
  * **Kelas Hambatan Samping (KHS):** Sangat Rendah, Rendah, Sedang, Tinggi, Sangat Tinggi.
  * **Ukuran Kota / Populasi:** $< 0.1\text{ Jt}$, $0.1 - 0.5\text{ Jt}$, $0.5 - 1.0\text{ Jt}$, $1.0 - 3.0\text{ Jt}$, $> 3.0\text{ Jt}$ ($FC_{UK}$).
  * **Pemisahan Arah (PA):** 50-50, 55-45, 60-40, 65-35, 70-30 ($FC_{PA}$).
* **Perhitungan Kapasitas & Kinerja Otomatis:**
  $$\begin{aligned}
  Q &= \sum (\text{Volume Kendaraan} \times \text{EMP}) \quad [\text{smp/jam}] \\
  C &= C_0 \times FC_{LJ} \times FC_{PA} \times FC_{HS} \times FC_{UK} \quad [\text{smp/jam}] \\
  DS &= \frac{Q}{C}
  \end{aligned}$$
* **Penentuan Level of Service (LOS):**
  * **LOS A** ($DS \le 0.20$): Bebas / Sangat Lancar
  * **LOS B** ($0.21 - 0.44$): Arus Stabil / Lancar
  * **LOS C** ($0.45 - 0.74$): Arus Stabil / Terkendali
  * **LOS D** ($0.75 - 0.84$): Mendekati Arus Tak Stabil
  * **LOS E** ($0.85 - 1.00$): Arus Kritis / Jenuh
  * **LOS F** ($DS > 1.00$): Macet Total / *Breakdown*
* **Visualisasi Data Interaktif:**
  * 4 Kartu KPI Metrik Modern (Arus $Q$, Kapasitas $C$, Derajat Kejenuhan $DS$, Badge LOS).
  * Diagram Batang Komparasi $Q$ vs $C$ dengan garis batas kritis $DS = 0.85$.
  * Donut Chart proporsi armada kendaraan (% SM, MP, KS, BB, TB).
  * Ekspor laporan hasil analisis ke format CSV siap cetak.

---

### 🎥 TAB 2: Smart CCTV Monitor & AI Vehicle Tracking
* **Mesin Deteksi Objek Ultralytics YOLO11:**
  * Pilihan arsitektur: `yolo11n.pt` (Ultra Cepat), `yolo11s.pt` (Akurasi Tinggi), dan `yolov8n.pt`.
* **Universal 4-Layer CCTV Stream Resolver:**
  1. **Lapis 1 (YouTube Live Dishub via yt-dlp):** Mengekstrak siaran langsung CCTV YouTube Live dari Dishub mana pun di Indonesia (misal: Bandung, DIY, Banyumas, dsb.).
  2. **Lapis 2 (Direct Stream):** Stream langsung `.m3u8`, `.mp4`, `.ts`, `.mjpg`, serta protokol `rtsp://` dan `rtmp://`.
  3. **Lapis 3 (Universal Web & Iframe Scraper):** Ekstraksi otomatis dari web portal daerah (`.html`) dan Restreamer (`/memfs/{uuid}.m3u8`).
  4. **Lapis 4 (Fallback HLS Playlist Reader):** Penembus restriksi SSL TLS tanpa crash pada OpenCV.
  5. **Preset Siap Pakai:** Lebih dari 40 titik kamera CCTV aktif Kota Magelang (RSJ Utara, Alun-Alun Gardena, Simpang GOR, dsb.) serta opsi Webcam lokal (`0`).
* **Optimasi Malam Hari & Aspal Basah:**
  * **Slider Confidence:** Default $0.20$ ($0.10 - 0.80$) untuk menangkap siluet minim cahaya.
  * **CLAHE Auto-Enhance:** Peningkatan kontras adaptif (*Contrast Limited Adaptive Histogram Equalization*) pada kanal L (LAB) sebelum inferensi YOLO11.
  * **Dukungan Roda Dua COCO:** Memetakan class 1 (*bicycle*) dan class 3 (*motorcycle*) ke kategori **"SM (Sepeda Motor)"**.
* **Logika Garis Hitung Virtual (*Anti-Miss Crossing* & *Neon Green Flash*):**
  * Pelacakan koordinat multi-frame berbasis `first_y` dan `last_y` untuk mencegah kendaraan terlewat saat terjadi oklusi sesaat.
  * Deteksi arah otomatis (*Directional Split* Arah 1 vs Arah 2) untuk kalibrasi otomatis faktor $FC_{PA}$.
  * Animasi garis berubah warna menjadi **HIJAU NEON TEBAL (4 px)** lengkap dengan badge pop-up `CROSSED: +1 SM (#ID)` selama 10 frame saat kendaraan melintas.
* **Sinkronisasi 1-Klik:**
  * Mengonversi hitungan durasi video aktual menjadi proyeksi volume arus per jam ($Q$).
  * Tombol *"🚀 Terapkan Hasil Hitungan CCTV ke Tab 1"* untuk langsung menyalurkan hasil hitungan AI ke kalkulator PKJI 2023.

---

## 🛠️ Persiapan & Struktur File Repository

```text
├── app.py                  # Script utama aplikasi web Streamlit
├── requirements.txt        # Dependensi Python cloud-optimized (CPU PyTorch + OpenCV Headless)
├── packages.txt            # Dependensi library sistem Linux Debian/Ubuntu (ffmpeg, libgl1, libglib2.0-0)
├── .gitignore              # Mengabaikan file cache, venv, dan temporary recording
├── sample_traffic.mp4      # Video sampel lalu lintas perkotaan untuk demo pengujian
├── run_app.bat             # Launcher 1-klik untuk pengguna Windows
├── run.ps1                 # Script PowerShell alternatif
└── README.md               # Dokumentasi lengkap proyek
```

---

## ☁️ Panduan Deploy ke Streamlit Community Cloud

Aplikasi ini sudah dipersiapkan secara khusus (*cloud-ready*) agar tidak mengalami error `libGL.so.1` atau kehabisan memori (*Out Of Memory*) di server Linux Streamlit Cloud:

1. **Fork atau Push** repository ini ke akun GitHub Anda:
   `https://github.com/Kaesaaa/TransMetrics-PKJI2023`
2. Buka **[share.streamlit.io](https://share.streamlit.io/)** dan login menggunakan akun GitHub Anda.
3. Klik tombol **"New app"**.
4. Isi parameter deployment:
   * **Repository:** `Kaesaaa/TransMetrics-PKJI2023`
   * **Branch:** `main`
   * **Main file path:** `app.py`
5. Klik **"Deploy!"**
6. Server Streamlit Cloud akan membaca `packages.txt` dan `requirements.txt` secara otomatis. Aplikasi akan online dan dapat diakses dari browser maupun smartphone secara publik!

---

## 💻 Panduan Menjalankan di Komputer Lokal (Windows)

### Cara 1: Menggunakan File Batch (1-Klik Praktis)
Cukup klik dua kali berkas:
```bash
run_app.bat
```

### Cara 2: Lewat Terminal PowerShell
```powershell
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍🎓 Penulis & Informasi Akademik

* **Program Studi:** S1 Teknik Sipil
* **Fokus:** Rekayasa Lalu Lintas & Sistem Transportasi Cerdas (*Intelligent Transportation Systems / ITS*)
* **Mata Kuliah:** Aplikasi Komputer / Rekayasa Transportasi
* **Pedoman Acuan:** Pedoman Kapasitas Jalan Indonesia (PKJI 2023) — Kementerian PUPR Ditjen Bina Marga

---
*Dibuat dengan dedikasi untuk kemajuan digitalisasi pemantauan lalu lintas dan rekayasa transportasi Indonesia.* 🇮🇩
