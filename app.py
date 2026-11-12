"""
===================================================================================================
TRANS-METRICS: Kalkulator Kinerja Ruas Jalan PKJI 2023 & Smart CCTV Monitor
Aplikasi Web Interaktif Rekayasa Lalu Lintas & Pemantauan Cerdas Dishub
Pedoman Kapasitas Jalan Indonesia (PKJI 2023) - Kementerian PUPR Ditjen Bina Marga
Terintegrasi Computer Vision: OpenCV, Ultralytics YOLO11, & ByteTrack Multi-Object Tracking
===================================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import sys
import subprocess
import tempfile
import cv2
import re
import urllib.parse
import urllib3
import requests
from bs4 import BeautifulSoup

try:
    import yt_dlp
except ImportError:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], check=False)
        import yt_dlp
    except Exception:
        yt_dlp = None

from ultralytics import YOLO

# Nonaktifkan peringatan SSL tidak aman (insecure request warnings)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Opsi FFmpeg OpenCV untuk bypass verifikasi TLS & timeout jaringan
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "tls_verify;0|timeout;10000000"

# -------------------------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="TRANS-METRICS | PKJI 2023 & Smart CCTV YOLO11",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------------------------------
# 2. CUSTOM CSS - TAMPILAN MODERN & PROFESIONAL TEKNIK SIPIL TRANSPORTASI
# -------------------------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Top Brand Banner */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        margin-bottom: 24px;
        border-left: 6px solid #0284c7;
        position: relative;
        overflow: hidden;
    }
    
    .brand-header::after {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 140px;
        height: 140px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }

    .brand-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-subtitle {
        font-size: 13.5px;
        color: #94a3b8;
        margin-top: 6px;
        font-weight: 400;
        line-height: 1.5;
    }

    .brand-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .tech-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(4px);
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    @media (max-width: 900px) {
        .kpi-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 600px) {
        .kpi-container {
            grid-template-columns: 1fr;
        }
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-unit {
        font-size: 13px;
        font-weight: 500;
        color: #64748b;
        margin-left: 4px;
    }

    .kpi-subtext {
        font-size: 11.5px;
        color: #94a3b8;
        margin-top: 8px;
        font-weight: 500;
    }

    /* LOS Badge Styles */
    .los-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 20px;
        letter-spacing: 0.5px;
        color: #ffffff;
        width: 100%;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }

    .los-desc {
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Content Cards */
    .content-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .content-box-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 10px;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        color: #475569;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0284c7 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08) !important;
    }

    .info-callout {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #0369a1;
        margin-bottom: 14px;
    }

    /* Dark Mode Compatibility */
    @media (prefers-color-scheme: dark) {
        .kpi-card, .content-box {
            background: #1e293b;
            border-color: #334155;
        }
        .kpi-value {
            color: #f8fafc;
        }
        .content-box-title {
            color: #f8fafc;
            border-bottom-color: #334155;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e293b;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0f172a !important;
            color: #38bdf8 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 3. INISIALISASI SESSION STATE LENGKAP
# -------------------------------------------------------------------------------------------------
if "vol_sm" not in st.session_state:
    st.session_state.vol_sm = 1850
if "vol_mp" not in st.session_state:
    st.session_state.vol_mp = 720
if "vol_ks" not in st.session_state:
    st.session_state.vol_ks = 110
if "vol_bb" not in st.session_state:
    st.session_state.vol_bb = 45
if "vol_tb" not in st.session_state:
    st.session_state.vol_tb = 35
if "pemisahan_arah" not in st.session_state:
    st.session_state.pemisahan_arah = "50-50"
if "cctv_applied" not in st.session_state:
    st.session_state.cctv_applied = False
if "cctv_applied_total" not in st.session_state:
    st.session_state.cctv_applied_total = 0
if "cctv_applied_pa" not in st.session_state:
    st.session_state.cctv_applied_pa = "50-50"

# State untuk hitungan live CCTV & Directional Split
if "cctv_counts" not in st.session_state:
    st.session_state.cctv_counts = {"SM": 0, "MP": 0, "KS": 0, "BB": 0, "TB": 0}
if "cctv_dir_counts" not in st.session_state:
    st.session_state.cctv_dir_counts = {"Arah 1": 0, "Arah 2": 0}
if "cctv_elapsed_sec" not in st.session_state:
    st.session_state.cctv_elapsed_sec = 0.0
if "cctv_frame_count" not in st.session_state:
    st.session_state.cctv_frame_count = 0
if "cctv_video_fps" not in st.session_state:
    st.session_state.cctv_video_fps = 25.0
if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False
if "last_annotated_frame" not in st.session_state:
    st.session_state.last_annotated_frame = None

# -------------------------------------------------------------------------------------------------
# 4. PRESET LIVE CCTV KOTA MAGELANG, SMART WEB EXTRACTOR, & HLS STREAM ENGINE
# -------------------------------------------------------------------------------------------------
MAGELANG_CCTV_PRESETS = {
    "RSJ - Utara (Jalan Utama / Rekomendasi)": "67338104-0028-4720-9f47-110fd4c0e7a9",
    "RSJ - Selatan": "c73a535b-71d2-4ac3-b1ab-c92ca9d2adc9",
    "Alun\" - GARDENA (Pusat Kota)": "8fb93fb3-cf14-4d29-8ec5-6ae8d56419df",
    "Alun\" - 'MAGELANG'": "2cf96a42-3def-4d0e-b38d-973173d011b7",
    "Alun\" - PTZ Tengah": "3ae4cf0b-df1b-405b-be6a-164bba2814f2",
    "Simpang GOR - Barat": "99d847b9-dd05-476a-bf55-b8a06251ec8c",
    "Simpang GOR - SELATAN": "2a2f0817-5e07-40f0-ac7e-3ae7811671db",
    "Simpang GOR - Timur": "198b13c0-f995-471a-8c56-5899cd1f55cd",
    "PLENGKUNG - TIMUR (SMKN 3)": "4492f5d9-be05-4fd5-a536-4849d5582136",
    "Plengkung - Barat (SMKN 3)": "79f5ca33-6a6d-4a4a-98d3-838908afb562",
    "Dinsos - Selatan": "e04df6dc-17d1-4a29-b1d2-0d7ba68f0a64",
    "Dinsos - Timur": "75adf7d7-6905-4377-b403-931d73d29033",
    "SMK Kristen 1 - Timur": "c6157074-9430-41e1-8cb7-72c99fa9ec59",
    "SMK Kristen 1 - Pertigaan": "b8266d83-4c6c-418f-a161-84cd07aa0b31",
    "SMPN1 - Utara": "2e22bf14-2428-483a-ae99-1d87f0973e29",
    "SMPN1 -  Selatan": "41299267-1f2d-413e-abf2-d55c1ee99dba",
    "SMPN3 - Timur": "fef8e680-46fe-4905-af36-69d84aebd4e1",
    "SMPN3 - Selatan": "8fe525a2-3291-494e-813d-3825a3d2e244",
    "SMPN3 - Pertigaan": "5d5acb18-c099-49a0-8c8b-e444b9b568bf",
    "Perempatan Terminal C": "68267420-6aef-49ef-9306-f10b1f90d2ba",
    "Pertigaan BNI": "da9c4d13-77a6-435b-a46c-ee1d457f8ac8",
    "BATAS KOTA": "7df59214-0e5b-40df-b467-75a7857b0159",
    "PSC - Utara": "090e1664-df75-410d-a4ea-b0db0d334329",
    "PSC - Selatan": "342540b3-545c-41a4-b3a2-624f7d824cdc",
    "Rumdin DPRD - Selatan": "d0133ed3-fc7d-4738-86fb-718df392f2ab",
    "Rumdin DPRD - Utara": "46b6bb7b-7187-480d-81a1-cd1f41a4c128",
    "Pusk Wates - Utara": "540a5d5a-c3da-4a7d-a0f9-caaf156e34e0",
    "Pusk Wates - Selatan": "5b1e52a8-cb29-4f3c-8916-35b5d78962ef",
    "Pusk Mgl Selatan - Utara": "00a237d2-bae2-4d08-9c0d-ab06986262ff",
    "Pusk Mgl Selatan - Timur": "a000ece4-361d-400c-9602-7fa74cd9894e",
    "Pusk Mgl Selatan - Barat": "4c3dac48-9b48-482f-9f84-c10f14ed471e",
    "Taman Badaan - Jl. Utara": "f5e78afa-c429-4ab1-b31e-e4b3b637e5d7",
    "Taman Remaja Magersari - Pertigaan": "af21cfe1-5574-418c-97f1-d7b779fe6d39",
    "Tower Air - Telkom": "445bc62e-35b5-4bb3-802c-2507bd78c035",
    "Simpang Satpol - Barat": "54f6ec38-e985-4186-addb-fa471f1c9d56",
    "Simpang Satpol - Selatan": "a1aa0214-9e0b-477c-8264-fdad75a0528d",
    "Simpang Satpol - Timur - Utara": "031a639e-1e4b-478c-af0c-70ccac058fa2",
    "Sumber Waras - Selatan": "d5400625-14f3-4787-8d57-80ea84b77154",
    "Sumber Waras - Utara": "115734d1-b3f5-42c4-b57c-0444bad14cb8"
}

if "magelang_channels" not in st.session_state:
    st.session_state.magelang_channels = dict(MAGELANG_CCTV_PRESETS)

def normalize_youtube_url(url: str) -> str:
    """
    Menormalisasi berbagai variasi link YouTube Live / Shorts / Embed
    menjadi format standar watch?v= agar mudah diekstrak oleh yt-dlp.
    """
    if not url:
        return ""
    m_live = re.search(r'youtube\.com/live/([a-zA-Z0-9_\-]+)', url)
    if m_live:
        return f"https://www.youtube.com/watch?v={m_live.group(1)}"
    m_short = re.search(r'youtu\.be/([a-zA-Z0-9_\-]+)', url)
    if m_short:
        return f"https://www.youtube.com/watch?v={m_short.group(1)}"
    m_embed = re.search(r'youtube\.com/embed/([a-zA-Z0-9_\-]+)', url)
    if m_embed:
        return f"https://www.youtube.com/watch?v={m_embed.group(1)}"
    return url


def extract_stream_ytdlp(url: str):
    """
    Lapis 2: Ekstraksi stream video dari YouTube Live atau portal video via yt-dlp.
    Mengembalikan: (stream_url, title, error_message)
    """
    if yt_dlp is None:
        return None, None, "Library yt-dlp belum terpasang."

    clean_url = normalize_youtube_url(url)
    ydl_opts = {
        'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'ios', 'web']}},
        'socket_timeout': 10,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
            if not info:
                return None, None, "Tidak dapat membaca metadata YouTube."
            title = info.get('title', 'YouTube Live Stream')
            stream_url = info.get('url')

            # Jika stream_url kosong/None, cari di formats (resolusi <= 720p dengan vcodec dan protocol m3u8)
            if not stream_url:
                formats = info.get('formats', [])
                valid_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('url')]
                m3u8_formats = [f for f in valid_formats if 'm3u8' in str(f.get('protocol', '')) or '.m3u8' in str(f.get('url', ''))]
                if m3u8_formats:
                    stream_url = m3u8_formats[-1]['url']
                elif valid_formats:
                    stream_url = valid_formats[-1]['url']

            if stream_url:
                return stream_url, title, None
            return None, title, "Tidak ditemukan format video yang dapat diputar."
    except Exception as e:
        return None, None, str(e)


def resolve_cctv_url(raw_input: str):
    """
    Mengekstrak dan mengubah input URL CCTV dari kota mana pun di Indonesia
    menjadi direct stream video/HLS (.m3u8/.mp4) dengan 4 Lapis Ekstraksi Universal.

    Returns:
        resolved_stream_url: str atau int (device webcam)
        camera_title: str
        extracted_channels: dict {nama_channel: uuid}
        status_msg: str
        is_success: bool
    """
    if raw_input is None:
        return None, "", {}, "", False

    raw_clean = str(raw_input).strip()
    if not raw_clean:
        return None, "", {}, "", False

    # Kasus device ID lokal (webcam)
    if raw_clean.isdigit():
        return int(raw_clean), f"Webcam Device {raw_clean}", {}, "Menggunakan Kamera Webcam Lokal", True

    # Jika file lokal (.mp4, .avi, etc)
    if os.path.exists(raw_clean):
        return raw_clean, os.path.basename(raw_clean), {}, "Berkas video lokal", True

    # ---------------------------------------------------------------------------------------------
    # LAPIS 1: YouTube Live & Portal Video Streaming (via yt-dlp)
    # Diperiksa lebih dahulu agar URL YouTube Live tidak tertukar dengan direct stream
    # ---------------------------------------------------------------------------------------------
    yt_patterns = ["youtube.com/", "youtu.be/", "youtube.com/live/", "youtube.com/embed/"]
    if any(p in raw_clean.lower() for p in yt_patterns):
        yt_stream, yt_title, yt_err = extract_stream_ytdlp(raw_clean)
        if yt_stream:
            return yt_stream, f"YouTube Live: {yt_title}", {}, f"[Lapis 1] Berhasil mengekstrak siaran YouTube Live: {yt_title}", True
        else:
            return raw_clean, "YouTube Stream (Gagal Ekstraksi)", {}, f"yt-dlp gagal: {yt_err or 'Format tidak ditemukan'}", False

    # ---------------------------------------------------------------------------------------------
    # LAPIS 2: Direct Stream (.m3u8, .mp4, .ts, .mjpg, .mjpeg, RTSP, RTMP)
    # ---------------------------------------------------------------------------------------------
    direct_exts = (".m3u8", ".mp4", ".ts", ".mjpg", ".mjpeg", ".avi", ".mkv")
    clean_no_query = raw_clean.lower().split("?")[0]
    is_direct_ext = any(clean_no_query.endswith(ext) for ext in direct_exts)
    is_direct_proto = raw_clean.startswith(("rtsp://", "rtmp://")) or (
        raw_clean.startswith(("http://", "https://")) and (
            ".m3u8" in raw_clean or is_direct_ext or "/memfs/" in raw_clean
        )
    )

    if is_direct_ext or (is_direct_proto and not clean_no_query.endswith((".html", ".htm", ".php", ".asp", ".aspx"))):
        return raw_clean, "Aliran Stream Langsung (Direct Stream)", {}, f"[Lapis 2] Direct stream terdeteksi: {raw_clean}", True

    # ---------------------------------------------------------------------------------------------
    # LAPIS 3: Universal HTML, Iframe & Restreamer Scraper untuk Web Pemkot / Dishub
    # ---------------------------------------------------------------------------------------------
    camera_title = "Live CCTV Kota"
    extracted_channels = {}
    resolved_m3u8 = None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    if raw_clean.startswith("http://") or raw_clean.startswith("https://"):
        try:
            resp = requests.get(raw_clean, headers=headers, verify=False, timeout=8)
            if resp.status_code == 200:
                html_text = resp.text
                soup = BeautifulSoup(html_text, 'html.parser')

                # Ekstrak Judul Halaman
                if soup.title and soup.title.string:
                    cand_title = soup.title.string.strip()
                    if cand_title and "Restreamer" not in cand_title and "404" not in cand_title:
                        camera_title = cand_title

                # a) Cari direct link .m3u8 atau .mp4 di dalam kode HTML
                m3u8_matches = re.findall(r'[\'"]([^\'"]+?\.(?:m3u8|mp4)(?:[^\'"]*?))[\'"]', html_text)
                if not m3u8_matches:
                    m3u8_matches = re.findall(r'(https?://[^\s"\'<>]+\.(?:m3u8|mp4)[^\s"\'<>]*)', html_text)

                if m3u8_matches:
                    resolved_m3u8 = urllib.parse.urljoin(raw_clean, m3u8_matches[0])

                # b) Cek Restreamer UUID (seperti dev.cctv.magelangkota.go.id atau pemkot lain pengguna Restreamer)
                uuid_match = re.search(r'([0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12})', raw_clean + " " + html_text)
                if resolved_m3u8 is None and uuid_match:
                    uuid_val = uuid_match.group(1)
                    parsed = urllib.parse.urlparse(raw_clean)
                    resolved_m3u8 = f"{parsed.scheme}://{parsed.netloc}/memfs/{uuid_val}.m3u8"

                # c) Periksa tag <iframe> (banyak web CCTV Pemkot menanamkan player di dalam iframe)
                if resolved_m3u8 is None:
                    for iframe in soup.find_all('iframe', src=True):
                        iframe_src = urllib.parse.urljoin(raw_clean, iframe['src'])
                        # Jika iframe adalah YouTube Live
                        if any(p in iframe_src.lower() for p in yt_patterns):
                            yt_stream, yt_title, _ = extract_stream_ytdlp(iframe_src)
                            if yt_stream:
                                return yt_stream, f"Embedded YouTube: {yt_title}", {}, f"[Lapis 3] Berhasil mengekstrak iframe YouTube Live: {yt_title}", True

                        # Coba telusuri isi iframe
                        try:
                            r_iframe = requests.get(iframe_src, headers=headers, verify=False, timeout=6)
                            if r_iframe.status_code == 200:
                                iframe_m3u8 = re.findall(r'[\'"]([^\'"]+?\.(?:m3u8|mp4)(?:[^\'"]*?))[\'"]', r_iframe.text)
                                if iframe_m3u8:
                                    resolved_m3u8 = urllib.parse.urljoin(iframe_src, iframe_m3u8[0])
                                    break
                                iframe_uuid = re.search(r'([0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12})', r_iframe.text)
                                if iframe_uuid:
                                    parsed_if = urllib.parse.urlparse(iframe_src)
                                    resolved_m3u8 = f"{parsed_if.scheme}://{parsed_if.netloc}/memfs/{iframe_uuid.group(1)}.m3u8"
                                    break
                        except Exception:
                            pass

                # d) Ekstrak tautan channel kamera lain (auto-scraper channel CCTV)
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    m = re.search(r'playersite_([0-9a-fA-F\-]{36})\.html', href)
                    if m:
                        u = m.group(1)
                        raw_t = a.get_text(separator=" ", strip=True)
                        c_name = re.sub(r'^LIVE\s*', '', raw_t, flags=re.IGNORECASE)
                        c_name = re.sub(r'Views.*$', '', c_name, flags=re.IGNORECASE).strip()
                        if c_name:
                            extracted_channels[c_name] = u

        except Exception:
            pass

    if resolved_m3u8:
        return resolved_m3u8, camera_title, extracted_channels, f"[Lapis 3] Berhasil mengekstrak aliran HLS dari web portal: {resolved_m3u8}", True

    # Fallback terakhir via yt-dlp jika web memakai video player publik yang dikenali yt-dlp
    if raw_clean.startswith("http://") or raw_clean.startswith("https://"):
        try:
            alt_stream, alt_title, _ = extract_stream_ytdlp(raw_clean)
            if alt_stream:
                return alt_stream, alt_title or camera_title, extracted_channels, f"[Lapis 2] Aliran video diekstrak via media streamer: {alt_title}", True
        except Exception:
            pass

    return raw_clean, camera_title, extracted_channels, "Menggunakan URL mentah", False


class HLSPlaylistReader:
    """
    Pembaca HLS Stream Fallback anti-gagal menggunakan requests (verify=False).
    Mendukung Master Playlist (#EXT-X-STREAM-INF) dan pengunduhan segmen .ts berkala
    baik URL path relatif maupun URL penuh (googlevideo/YouTube/CCTV Pemkot).
    """
    def __init__(self, m3u8_url):
        self.m3u8_url = m3u8_url
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.seen_segments = set()
        self.current_cap = None
        self.temp_ts_path = os.path.abspath(f"temp_hls_{os.getpid()}.ts")
        self.is_opened = True

    def _resolve_media_playlist(self):
        try:
            r = self.session.get(self.m3u8_url, timeout=6)
            if r.status_code != 200:
                return self.m3u8_url
            text = r.text
            # Jika mengandung #EXT-X-STREAM-INF, ini adalah Master Playlist
            if "#EXT-X-STREAM-INF" in text:
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                sub_candidates = []
                for idx, line in enumerate(lines):
                    if line.startswith("#EXT-X-STREAM-INF") and (idx + 1) < len(lines):
                        cand_url = lines[idx + 1].strip()
                        if not cand_url.startswith("#"):
                            sub_candidates.append(cand_url)
                if sub_candidates:
                    return urllib.parse.urljoin(self.m3u8_url, sub_candidates[-1])
            return self.m3u8_url
        except Exception:
            return self.m3u8_url

    def _fetch_next_segment(self):
        playlist_url = self._resolve_media_playlist()
        if not playlist_url:
            return None
        try:
            r = self.session.get(playlist_url, timeout=6)
            if r.status_code != 200:
                return None
            lines = [l.strip() for l in r.text.splitlines() if l.strip()]
            segments = []
            for l in lines:
                if not l.startswith("#"):
                    full_seg_url = urllib.parse.urljoin(playlist_url, l)
                    segments.append(full_seg_url)

            new_segments = [s for s in segments if s not in self.seen_segments]
            if not new_segments and segments:
                new_segments = [segments[-1]]

            if new_segments:
                seg_url = new_segments[-1]
                self.seen_segments.add(seg_url)
                if len(self.seen_segments) > 40:
                    self.seen_segments = set(list(self.seen_segments)[-15:])

                seg_resp = self.session.get(seg_url, timeout=7)
                if seg_resp.status_code == 200 and len(seg_resp.content) > 1000:
                    with open(self.temp_ts_path, "wb") as f:
                        f.write(seg_resp.content)
                    return self.temp_ts_path
        except Exception:
            pass
        return None

    def read(self):
        for _ in range(3):
            if self.current_cap is not None and self.current_cap.isOpened():
                ret, frame = self.current_cap.read()
                if ret and frame is not None:
                    return True, frame
                else:
                    self.current_cap.release()
                    self.current_cap = None

            ts_path = self._fetch_next_segment()
            if ts_path and os.path.exists(ts_path):
                self.current_cap = cv2.VideoCapture(ts_path)
                ret, frame = self.current_cap.read()
                if ret and frame is not None:
                    return True, frame
            time.sleep(0.2)
        return False, None

    def isOpened(self):
        return self.is_opened

    def release(self):
        self.is_opened = False
        if self.current_cap is not None:
            self.current_cap.release()
            self.current_cap = None
        if os.path.exists(self.temp_ts_path):
            try:
                os.remove(self.temp_ts_path)
            except Exception:
                pass


class SmartVideoSource:
    """
    Abstraksi Universal Video Source yang mendukung Webcam, Video File (.mp4/.avi),
    Direct RTSP/HLS FFmpeg, serta Fallback HLS Playlist Reader jika terkendala SSL.
    """
    def __init__(self, source):
        self.source = source
        self.is_hls = False
        self.cap = None
        self.hls_reader = None
        self.frame_w = 800
        self.frame_h = 480
        self.fps = 25.0
        self._init_source()

    def _init_source(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "tls_verify;0|timeout;10000000"

        if isinstance(self.source, str) and (".m3u8" in self.source or "memfs" in self.source or "googlevideo" in self.source or "/hls" in self.source):
            self.is_hls = True
            direct_cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if direct_cap.isOpened():
                ret, test_frame = direct_cap.read()
                if ret and test_frame is not None:
                    self.cap = direct_cap
                    self.frame_w = int(direct_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or test_frame.shape[1]
                    self.frame_h = int(direct_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or test_frame.shape[0]
                    self.fps = direct_cap.get(cv2.CAP_PROP_FPS) or 25.0
                    return
                direct_cap.release()

            # Fallback ke HLSPlaylistReader
            self.hls_reader = HLSPlaylistReader(self.source)
            ret, test_frame = self.hls_reader.read()
            if ret and test_frame is not None:
                self.frame_h, self.frame_w = test_frame.shape[:2]
                self.fps = 25.0
                return
        else:
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 800
                self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fps_val = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = fps_val if (fps_val and fps_val > 0 and not np.isnan(fps_val)) else 25.0

    def isOpened(self):
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.hls_reader is not None and self.hls_reader.isOpened():
            return True
        return False

    def read(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
            if self.is_hls:
                self.cap.release()
                self.cap = None
                self.hls_reader = HLSPlaylistReader(self.source)
                return self.hls_reader.read()
            return False, None

        if self.hls_reader is not None and self.hls_reader.isOpened():
            return self.hls_reader.read()

        return False, None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hls_reader is not None:
            self.hls_reader.release()
            self.hls_reader = None


@st.cache_resource
def load_yolo_model(model_name: str):
    """
    Memuat model Ultralytics YOLO dengan cache resource agar tidak di-load berulang kali.
    """
    return YOLO(model_name)

def ensure_sample_video():
    """
    Memastikan file sample_traffic.mp4 tersedia secara lokal.
    """
    sample_file = "sample_traffic.mp4"
    if os.path.exists(sample_file) and os.path.getsize(sample_file) > 100000:
        return sample_file

    import urllib.request
    try:
        url = "https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/Traffic.mp4"
        urllib.request.urlretrieve(url, sample_file)
        return sample_file
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(sample_file, fourcc, 25.0, (800, 480))
        for f_idx in range(150):
            frame = np.full((480, 800, 3), 45, dtype=np.uint8)
            cv2.line(frame, (400, 0), (400, 480), (255, 255, 255), 3)
            y_pos = (f_idx * 6) % 520 - 40
            cv2.rectangle(frame, (430, y_pos), (520, y_pos + 60), (34, 197, 94), -1)
            out.write(frame)
        out.release()
        return sample_file

def apply_clahe_enhancement(bgr_image, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Peningkatan kontras adaptif untuk kondisi malam hari / hujan / silau lampu
    menggunakan Contrast Limited Adaptive Histogram Equalization (CLAHE) pada kanal Lightness (L) LAB.
    """
    try:
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr_image

def get_video_preview(video_source, line_pos_pct, enable_clahe=True):
    """
    Mengambil frame pertama sebagai thumbnail standby beserta garis hitung virtual.
    Mendukung webcam, video lokal, dan stream HLS/CCTV live.
    """
    if video_source is None:
        return None
    try:
        cap = SmartVideoSource(video_source)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                if enable_clahe:
                    frame = apply_clahe_enhancement(frame)
                h, w = frame.shape[:2]
                line_y = int(h * (line_pos_pct / 100.0))
                annotated = frame.copy()
                # Garis hitung virtual
                cv2.line(annotated, (0, line_y), (w, line_y), (0, 0, 255), 3)
                cv2.putText(annotated, f"--- VIRTUAL COUNTING LINE (ROI {line_pos_pct}%) ---", 
                            (20, max(28, line_y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                # HUD Top Bar
                cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
                cv2.putText(annotated, f"STANDBY | KLIK 'MULAI DETEKSI' UNTUK MEMPROSES VIDEO | RES: {w}x{h}", 
                            (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 2)
                return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None

def hitung_rekomendasi_pa(c1: int, c2: int):
    """
    Menghitung proporsi pemisahan arah dan merekomendasikan faktor PA PKJI 2023.
    """
    total = c1 + c2
    if total == 0:
        return "50-50", 50.0, 50.0
    p1 = (c1 / total) * 100.0
    p2 = (c2 / total) * 100.0
    p_max = max(p1, p2)
    if p_max <= 52.5:
        rec = "50-50"
    elif p_max <= 57.5:
        rec = "55-45"
    elif p_max <= 62.5:
        rec = "60-40"
    elif p_max <= 67.5:
        rec = "65-35"
    else:
        rec = "70-30"
    return rec, p1, p2

# -------------------------------------------------------------------------------------------------
# 5. TABEL EMPIRIS & LOGIKA PERHITUNGAN PKJI 2023
# -------------------------------------------------------------------------------------------------
def hitung_fc_lj(tipe_jalan: str, lebar_jalur: float) -> float:
    if "2/2 TT" in tipe_jalan:
        lebar_ref = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        fc_ref    = [0.80, 0.88, 0.92, 0.96, 1.00, 1.05, 1.11, 1.16, 1.21, 1.29, 1.34, 1.39, 1.44, 1.48]
        return float(np.interp(lebar_jalur, lebar_ref, fc_ref))
    else:
        lebar_lajur = lebar_jalur / 2.0
        lebar_lajur_ref = [2.75, 3.00, 3.25, 3.50, 3.75, 4.00]
        fc_ref          = [0.89, 0.92, 0.96, 1.00, 1.04, 1.08]
        return float(np.interp(lebar_lajur, lebar_lajur_ref, fc_ref))

def hitung_fc_pa(tipe_jalan: str, pemisahan_arah: str) -> float:
    if "4/2 T" in tipe_jalan:
        return 1.00

    tabel_pa_22 = {
        "50-50": 1.00,
        "55-45": 0.97,
        "60-40": 0.94,
        "65-35": 0.91,
        "70-30": 0.88
    }

    tabel_pa_42tt = {
        "50-50": 1.00,
        "55-45": 0.985,
        "60-40": 0.970,
        "65-35": 0.955,
        "70-30": 0.940
    }

    if "2/2 TT" in tipe_jalan:
        return tabel_pa_22.get(pemisahan_arah, 1.00)
    else:
        return tabel_pa_42tt.get(pemisahan_arah, 1.00)

def hitung_fc_hs(tipe_jalan: str, khs: str, lebar_bahu: float) -> float:
    bahu_ref = [0.5, 1.0, 1.5, 2.0]

    matriks_22 = {
        "Sangat Rendah": [0.96, 0.99, 1.01, 1.03],
        "Rendah":        [0.91, 0.95, 0.98, 1.00],
        "Sedang":        [0.86, 0.90, 0.95, 0.98],
        "Tinggi":        [0.78, 0.83, 0.88, 0.93],
        "Sangat Tinggi": [0.72, 0.77, 0.81, 0.86]
    }

    matriks_42 = {
        "Sangat Rendah": [0.99, 1.01, 1.03, 1.04],
        "Rendah":        [0.96, 0.98, 1.01, 1.03],
        "Sedang":        [0.91, 0.94, 0.97, 1.00],
        "Tinggi":        [0.84, 0.87, 0.91, 0.95],
        "Sangat Tinggi": [0.77, 0.81, 0.85, 0.90]
    }

    if "2/2 TT" in tipe_jalan:
        fc_list = matriks_22.get(khs, [0.86, 0.90, 0.95, 0.98])
    else:
        fc_list = matriks_42.get(khs, [0.91, 0.94, 0.97, 1.00])

    return float(np.interp(lebar_bahu, bahu_ref, fc_list))

def hitung_fc_uk(ukuran_kota: str) -> float:
    tabel_uk = {
        "< 0.1 Juta": 0.86,
        "0.1 - 0.5 Juta": 0.90,
        "0.5 - 1.0 Juta": 0.94,
        "1.0 - 3.0 Juta": 1.00,
        "> 3.0 Juta": 1.04
    }
    return tabel_uk.get(ukuran_kota, 1.00)

def evaluasi_los(ds: float):
    if ds <= 0.20:
        return {
            "los": "A", "status": "Sangat Lancar",
            "keterangan": "Kondisi arus bebas, kecepatan ditentukan oleh batas maksimum jalan.",
            "warna": "#15803d", "bg_color": "#dcfce7", "text_color": "#14532d",
            "rekomendasi": "Pertahankan kondisi geometrik & marka jalan saat ini. Tidak diperlukan rekayasa struktural."
        }
    elif ds <= 0.44:
        return {
            "los": "B", "status": "Arus Stabil / Lancar",
            "keterangan": "Arus stabil, kecepatan operasi sedikit terbatas oleh interaksi lalu lintas.",
            "warna": "#22c55e", "bg_color": "#f0fdf4", "text_color": "#166534",
            "rekomendasi": "Lakukan pemantauan berkala dan pengawasan parkir liar di bahu jalan."
        }
    elif ds <= 0.74:
        return {
            "los": "C", "status": "Arus Stabil / Terkendali",
            "keterangan": "Arus stabil tetapi kecepatan dan kemampuan bermanuver mulai dibatasi kendaraan lain.",
            "warna": "#eab308", "bg_color": "#fef9c3", "text_color": "#854d0e",
            "rekomendasi": "Zona ideal kapasitas jalan perkotaan. Perketat regulasi on-street parking dan titik penyeberangan."
        }
    elif ds <= 0.84:
        return {
            "los": "D", "status": "Mendekati Tak Stabil",
            "keterangan": "Arus mendekati tidak stabil, penurunan kecepatan signifikan saat volume meningkat.",
            "warna": "#f97316", "bg_color": "#ffedd5", "text_color": "#9a3412",
            "rekomendasi": "Peringatan dini (Early Warning). Perlu sterilisasi bahu jalan dan kanalisasi persimpangan."
        }
    elif ds <= 1.00:
        return {
            "los": "E", "status": "Arus Kritis / Jenuh",
            "keterangan": "Volume mendekati kapasitas maksimum jalan. Kecepatan sangat rendah dan rawan henti.",
            "warna": "#f43f5e", "bg_color": "#ffe4e6", "text_color": "#9f1239",
            "rekomendasi": "KONDISI KRITIS (Ambang Batas DS=0.85 Terlampaui). Terapkan manajemen lalu lintas, ganjil-genap, atau satu arah."
        }
    else:
        return {
            "los": "F", "status": "Macet Total / Breakdown",
            "keterangan": "Arus jenuh melampaui kapasitas. Kecepatan merayap hingga berhenti, antrean panjang terbentuk.",
            "warna": "#b91c1c", "bg_color": "#fee2e2", "text_color": "#7f1d1d",
            "rekomendasi": "KONDISI DARURAT KEMACETAN. Diperlukan intervensi Dishub/Polantas segera: buka tutup jalur dan pengalihan arus."
        }

# -------------------------------------------------------------------------------------------------
# 6. HEADER UTAMA APLIKASI
# -------------------------------------------------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div class="brand-title">
        <span style="font-size:32px;">🚦</span>
        TRANS-METRICS
    </div>
    <div style="font-size:18px; font-weight:700; color:#38bdf8; margin-top:2px;">
        Kalkulator Kinerja Ruas Jalan PKJI 2023 & Smart CCTV Monitor
    </div>
    <div class="brand-subtitle">
        Sistem Perhitungan Kapasitas Ruas Jalan Perkotaan Terpadu Pedoman Kapasitas Jalan Indonesia (PKJI 2023)
        Kementerian PUPR Ditjen Bina Marga dilengkapi Pemrosesan Video Real-Time AI YOLO11 & Pelacak ByteTrack.
    </div>
    <div class="brand-badges">
        <span class="tech-pill">📘 PKJI 2023 Standard</span>
        <span class="tech-pill">⚡ Ultralytics YOLO11 (yolo11n / yolo11s)</span>
        <span class="tech-pill">🎯 ByteTrack MOT & Directional Split</span>
        <span class="tech-pill">📏 Virtual Counting Line (ROI)</span>
        <span class="tech-pill">📊 Level of Service (LOS A - F)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 7. SIDEBAR: PARAMETER GEOMETRIK & KONDISI LINGKUNGAN
# -------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""<div style="padding-bottom:12px; border-bottom:2px solid #e2e8f0; margin-bottom:16px;">
<h3 style="margin:0; font-size:18px; color:#0f172a; font-weight:800; display:flex; align-items:center; gap:8px;">
<span>⚙️</span> Parameter Geometrik
</h3>
<p style="margin:4px 0 0 0; font-size:12px; color:#64748b;">
Karakteristik Fisik & Lingkungan Ruas Jalan
</p>
</div>""", unsafe_allow_html=True)

    # 1. Tipe Jalan
    tipe_jalan = st.selectbox(
        "Tipe Jalan Perkotaan",
        options=[
            "2/2 TT (Dua Lajur Dua Arah Tak-Terbagi)",
            "4/2 T (Empat Lajur Dua Arah Terbagi)",
            "4/2 TT (Empat Lajur Dua Arah Tak-Terbagi)"
        ],
        index=0,
        help="Pilih penampang melintang ruas jalan perkotaan sesuai klasifikasi PKJI 2023."
    )

    if "2/2 TT" in tipe_jalan:
        c0_default = 2900
        c0_label = "2900 smp/jam (Dua Arah Total)"
    elif "4/2 T" in tipe_jalan:
        c0_default = 3300
        c0_label = "3300 smp/jam (Per Arah / Jalur)"
    else:
        c0_default = 3000
        c0_label = "3000 smp/jam (Per Arah)"

    st.caption(f"📌 **Kapasitas Dasar ($C_0$):** `{c0_label}`")
    st.markdown("---")

    # 2. Lebar Jalur Lalu Lintas
    lebar_jalur = st.slider(
        "Lebar Jalur Lalu Lintas ($W_L$ / $W_e$)",
        min_value=5.0, max_value=14.0, value=7.0, step=0.5, format="%.1f meter",
        help="Untuk 2/2 TT merupakan total lebar 2 arah. Untuk 4/2 T merupakan lebar per arah."
    )

    # 3. Lebar Bahu Efektif
    lebar_bahu = st.slider(
        "Lebar Bahu Efektif ($W_S$)",
        min_value=0.5, max_value=3.0, value=1.5, step=0.25, format="%.2f meter",
        help="Lebar ruang bebas sisi/bahu jalan beraspal/kerb yang efektif."
    )
    st.markdown("---")

    # 4. Kelas Hambatan Samping
    khs = st.selectbox(
        "Kelas Hambatan Samping (KHS)",
        options=["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
        index=2,
        help="Frekuensi kejadian samping per jam (pejalan kaki, parkir tepi jalan, kendaraan lambat)."
    )

    # 5. Ukuran Kota
    ukuran_kota = st.selectbox(
        "Ukuran Kota / Populasi Penduduk",
        options=["< 0.1 Juta", "0.1 - 0.5 Juta", "0.5 - 1.0 Juta", "1.0 - 3.0 Juta", "> 3.0 Juta"],
        index=3
    )

    # 6. Pemisahan Arah (PA) - Terhubung dengan session state hasil AI
    if "4/2 T" in tipe_jalan:
        pemisahan_arah = "50-50"
        st.selectbox("Pemisahan Arah (PA)", options=["50-50"], index=0, disabled=True,
                     help="Jalan terbagi dengan median memiliki arah independen (FC_PA = 1.00).")
    else:
        pa_options = ["50-50", "55-45", "60-40", "65-35", "70-30"]
        saved_pa = st.session_state.get("pemisahan_arah", "50-50")
        default_pa_idx = pa_options.index(saved_pa) if saved_pa in pa_options else 0
        pemisahan_arah = st.selectbox(
            "Pemisahan Arah (PA)",
            options=pa_options,
            index=default_pa_idx,
            help="Proporsi pembagian arus lalu lintas antara dua arah yang berlawanan."
        )
        st.session_state.pemisahan_arah = pemisahan_arah

    st.markdown("---")
    st.markdown("<p style='font-size:12px; font-weight:700; color:#475569; margin-bottom:6px;'>⚡ Preset Skenario Cepat:</p>", unsafe_allow_html=True)
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("🏙️ Arteri Kota", use_container_width=True):
            st.session_state.vol_sm = 2800
            st.session_state.vol_mp = 1100
            st.session_state.vol_ks = 180
            st.session_state.vol_bb = 60
            st.session_state.vol_tb = 50
            st.rerun()
    with c_p2:
        if st.button("🏪 Pasar Ramai", use_container_width=True):
            st.session_state.vol_sm = 2200
            st.session_state.vol_mp = 550
            st.session_state.vol_ks = 130
            st.session_state.vol_bb = 20
            st.session_state.vol_tb = 15
            st.rerun()

# -------------------------------------------------------------------------------------------------
# 8. TAB UTAMA APLIKASI
# -------------------------------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "📊 TAB 1: Kalkulator Kinerja Ruas Jalan (PKJI 2023)",
    "📹 TAB 2: Integrasi Smart CCTV Dishub (Real-Time AI)"
])

# =================================================================================================
# KONTEN TAB 1: KALKULATOR KINERJA RUAS JALAN (PKJI 2023)
# =================================================================================================
with tab1:
    if st.session_state.cctv_applied:
        st.markdown(f"""<div class="info-callout">
✅ <strong>Data Volume Berhasil Disinkronisasi dari Smart CCTV Dishub (YOLO11 Live)!</strong><br>
Total Arus: <strong>{st.session_state.cctv_applied_total:,} kend/jam</strong> • Pemisahan Arah (PA): <strong>{st.session_state.cctv_applied_pa}</strong> telah diperbarui secara otomatis.
</div>""", unsafe_allow_html=True)
        st.session_state.cctv_applied = False

    # 1. Panel Input Volume Kendaraan
    with st.container():
        st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>🚗</span> Input Volume Kendaraan Manual (Per Jam)
</div>
<p style="font-size:13px; color:#64748b; margin-top:-8px; margin-bottom:16px;">
Masukkan volume lalu lintas terhitung per jam berdasarkan survei pencacahan manual atau hasil sinkronisasi kamera CCTV AI.
</p>""", unsafe_allow_html=True)

        col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
        with col_v1:
            in_sm = st.number_input(
                "🛵 Sepeda Motor (SM)",
                min_value=0, max_value=25000,
                value=int(st.session_state.vol_sm),
                step=50
            )
            st.session_state.vol_sm = in_sm
        with col_v2:
            in_mp = st.number_input(
                "🚙 Mobil Penumpang (MP)",
                min_value=0, max_value=15000,
                value=int(st.session_state.vol_mp),
                step=25
            )
            st.session_state.vol_mp = in_mp
        with col_v3:
            in_ks = st.number_input(
                "🚐 Kendaraan Sedang (KS)",
                min_value=0, max_value=5000,
                value=int(st.session_state.vol_ks),
                step=10
            )
            st.session_state.vol_ks = in_ks
        with col_v4:
            in_bb = st.number_input(
                "🚌 Bus Besar (BB)",
                min_value=0, max_value=3000,
                value=int(st.session_state.vol_bb),
                step=5
            )
            st.session_state.vol_bb = in_bb
        with col_v5:
            in_tb = st.number_input(
                "🚛 Truk Berat (TB)",
                min_value=0, max_value=3000,
                value=int(st.session_state.vol_tb),
                step=5
            )
            st.session_state.vol_tb = in_tb

        with st.expander("⚙️ Parameter Nilai Ekivalen Mobil Penumpang (EMP PKJI 2023)"):
            emp_sm_default = 0.25 if ("4/2" in tipe_jalan) else 0.40
            c_emp1, c_emp2, c_emp3, c_emp4, c_emp5 = st.columns(5)
            with c_emp1:
                emp_sm = st.number_input("EMP SM", min_value=0.10, max_value=1.00, value=float(emp_sm_default), step=0.05, format="%.2f")
            with c_emp2:
                emp_mp = st.number_input("EMP MP", min_value=1.00, max_value=1.00, value=1.00, step=0.01, format="%.2f", disabled=True)
            with c_emp3:
                emp_ks = st.number_input("EMP KS", min_value=0.50, max_value=2.50, value=1.30, step=0.05, format="%.2f")
            with c_emp4:
                emp_bb = st.number_input("EMP BB", min_value=0.50, max_value=2.50, value=1.30, step=0.05, format="%.2f")
            with c_emp5:
                emp_tb = st.number_input("EMP TB", min_value=0.50, max_value=3.00, value=1.50, step=0.05, format="%.2f")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------------------------
    # 2. LOGIKA MATEMATIKA PERHITUNGAN PKJI 2023
    # ---------------------------------------------------------------------------------------------
    q_sm = in_sm * emp_sm
    q_mp = in_mp * emp_mp
    q_ks = in_ks * emp_ks
    q_bb = in_bb * emp_bb
    q_tb = in_tb * emp_tb
    q_total = q_sm + q_mp + q_ks + q_bb + q_tb
    total_kendaraan = in_sm + in_mp + in_ks + in_bb + in_tb

    fc_lj = hitung_fc_lj(tipe_jalan, lebar_jalur)
    fc_pa = hitung_fc_pa(tipe_jalan, pemisahan_arah)
    fc_hs = hitung_fc_hs(tipe_jalan, khs, lebar_bahu)
    fc_uk = hitung_fc_uk(ukuran_kota)

    c_kapasitas = c0_default * fc_lj * fc_pa * fc_hs * fc_uk
    ds = q_total / c_kapasitas if c_kapasitas > 0 else 0.0
    los_info = evaluasi_los(ds)

    # ---------------------------------------------------------------------------------------------
    # 3. EMPAT KARTU KPI METRIK UTAMA
    # ---------------------------------------------------------------------------------------------
    kpi_cards_html = f"""<div class="kpi-container">
<div class="kpi-card" style="border-top: 4px solid #0284c7;">
<div class="kpi-label">
<span>Arus Aktual (Q)</span>
<span>📈</span>
</div>
<div class="kpi-value">{q_total:,.1f}<span class="kpi-unit">smp/jam</span></div>
<div class="kpi-subtext">Total: {total_kendaraan:,} kend/jam terkonversi</div>
</div>
<div class="kpi-card" style="border-top: 4px solid #6366f1;">
<div class="kpi-label">
<span>Kapasitas Efektif (C)</span>
<span>🛣️</span>
</div>
<div class="kpi-value">{c_kapasitas:,.1f}<span class="kpi-unit">smp/jam</span></div>
<div class="kpi-subtext">C₀ = {c0_default:,} smp/jam • 4 Faktor Koreksi</div>
</div>
<div class="kpi-card" style="border-top: 4px solid {los_info['warna']};">
<div class="kpi-label">
<span>Derajat Kejenuhan (DS)</span>
<span>⚖️</span>
</div>
<div class="kpi-value" style="color:{los_info['warna']};">{ds:.3f}<span class="kpi-unit">Q / C</span></div>
<div class="kpi-subtext">Beban Ruas: {(ds*100):.1f}% dari kapasitas</div>
</div>
<div class="kpi-card" style="border-top: 4px solid {los_info['warna']};">
<div class="kpi-label">
<span>Tingkat Pelayanan (LOS)</span>
<span>🚦</span>
</div>
<div class="los-badge" style="background-color: {los_info['warna']};">
LOS {los_info['los']}
</div>
<div class="los-desc" style="color: {los_info['warna']};">
{los_info['status']}
</div>
</div>
</div>"""
    st.markdown(kpi_cards_html, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------------------------
    # 4. GRAFIK BATANG INTERAKTIF Q VS C & DONUT CHART KOMPOSISI ARMADA
    # ---------------------------------------------------------------------------------------------
    g_col1, g_col2 = st.columns([6, 4])

    with g_col1:
        st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>📊</span> Perbandingan Arus Aktual (Q) vs Kapasitas Jalan (C)
</div>""", unsafe_allow_html=True)

        ambang_kritis = 0.85 * c_kapasitas
        sisa_kapasitas = c_kapasitas - q_total

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Arus Aktual (Q)",
            x=["Arus Lalu Lintas (Q)"],
            y=[q_total],
            marker=dict(color=los_info["warna"], line=dict(color="#0f172a", width=1.5)),
            text=[f"<b>{q_total:,.1f} smp/jam</b><br>({total_kendaraan:,} kend/jam)"],
            textposition="auto",
            hovertemplate="<b>Arus Aktual (Q)</b>: %{y:,.1f} smp/jam<extra></extra>"
        ))
        fig_bar.add_trace(go.Bar(
            name="Kapasitas Efektif (C)",
            x=["Kapasitas Jalan (C)"],
            y=[c_kapasitas],
            marker=dict(color="#64748b", line=dict(color="#334155", width=1.5)),
            text=[f"<b>{c_kapasitas:,.1f} smp/jam</b><br>(Kapasitas Maks)"],
            textposition="auto",
            hovertemplate="<b>Kapasitas Efektif (C)</b>: %{y:,.1f} smp/jam<extra></extra>"
        ))

        fig_bar.add_shape(
            type="line",
            x0=-0.5, x1=1.5,
            y0=ambang_kritis, y1=ambang_kritis,
            line=dict(color="#dc2626", width=2.5, dash="dashdot"),
        )
        fig_bar.add_annotation(
            x=0.5, y=ambang_kritis,
            text=f"⚠️ Ambang Kritis Batas Jenuh DS = 0.85 ({ambang_kritis:,.0f} smp/jam)",
            showarrow=True, arrowhead=2, arrowcolor="#dc2626",
            ax=0, ay=-25,
            bgcolor="#fee2e2", bordercolor="#dc2626", borderwidth=1, borderpad=4,
            font=dict(size=11, color="#991b1b", family="Plus Jakarta Sans", weight="bold")
        )

        y_max = max(q_total, c_kapasitas) * 1.25 if max(q_total, c_kapasitas) > 0 else 1000
        fig_bar.update_layout(
            height=370,
            margin=dict(l=40, r=40, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(tickfont=dict(size=13, weight="bold")),
            yaxis=dict(title="Volume (smp / jam)", range=[0, y_max], gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if sisa_kapasitas >= 0:
            st.markdown(f"""<div style="font-size:12.5px; color:#166534; background:#f0fdf4; padding:8px 12px; border-radius:6px; border:1px solid #bbf7d0;">
🟢 <strong>Sisa Cadangan Kapasitas Ruas:</strong> {sisa_kapasitas:,.1f} smp/jam ({((sisa_kapasitas/c_kapasitas)*100):.1f}% ruang bebas tersedia).
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="font-size:12.5px; color:#991b1b; background:#fef2f2; padding:8px 12px; border-radius:6px; border:1px solid #fecaca;">
🔴 <strong>Defisit / Kelebihan Arus (Overcapacity):</strong> {abs(sisa_kapasitas):,.1f} smp/jam melampaui daya tampung maksimal ruas!
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with g_col2:
        st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>🥧</span> Komposisi Armada Kendaraan
</div>""", unsafe_allow_html=True)

        labels = ["Sepeda Motor (SM)", "Mobil Penumpang (MP)", "Kendaraan Sedang (KS)", "Bus Besar (BB)", "Truk Berat (TB)"]
        values_kend = [in_sm, in_mp, in_ks, in_bb, in_tb]
        colors = ["#0284c7", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values_kend,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="percent+label",
            textposition="inside",
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Volume: %{value:,} kend/jam<br>Persentase: %{percent}<extra></extra>"
        )])

        fig_donut.update_layout(
            height=370,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[dict(text=f"<b>Total</b><br>{total_kendaraan:,}<br><span style='font-size:10px;'>kend/jam</span>", x=0.5, y=0.5, font_size=13, showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown(f"""<div style="font-size:12px; color:#475569; background:#f8fafc; padding:8px 12px; border-radius:6px; border:1px solid #e2e8f0;">
🏍️ Proporsi SM: <strong>{((in_sm/total_kendaraan*100) if total_kendaraan>0 else 0):.1f}%</strong> | 🚗 Proporsi MP: <strong>{((in_mp/total_kendaraan*100) if total_kendaraan>0 else 0):.1f}%</strong>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------------------------
    # 5. TABEL RINCIAN FAKTOR & EVALUASI TEKNIS
    # ---------------------------------------------------------------------------------------------
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>📐</span> Rincian Faktor Penyesuaian Kapasitas (PKJI 2023)
</div>""", unsafe_allow_html=True)

        df_faktor = pd.DataFrame({
            "Parameter Koefisien": [
                "Kapasitas Dasar (C₀)",
                "Faktor Penyesuaian Lebar Jalur (FC_LJ)",
                "Faktor Penyesuaian Pemisahan Arah (FC_PA)",
                "Faktor Penyesuaian Hambatan Samping (FC_HS)",
                "Faktor Penyesuaian Ukuran Kota (FC_UK)"
            ],
            "Kondisi Input": [
                f"{tipe_jalan.split('(')[0].strip()}",
                f"{lebar_jalur:.1f} m",
                f"{pemisahan_arah}",
                f"{khs} (Bahu: {lebar_bahu:.2f} m)",
                f"{ukuran_kota} Penduduk"
            ],
            "Nilai Koefisien": [
                f"{c0_default} smp/jam",
                f"{fc_lj:.3f}",
                f"{fc_pa:.3f}",
                f"{fc_hs:.3f}",
                f"{fc_uk:.3f}"
            ]
        })
        st.dataframe(df_faktor, use_container_width=True, hide_index=True)
        st.markdown(f"""<div style="font-family:'JetBrains Mono', monospace; font-size:12px; background:#f1f5f9; padding:10px 12px; border-radius:8px; margin-top:8px; border:1px solid #cbd5e1;">
<strong>Kapasitas C = C₀ × FC_LJ × FC_PA × FC_HS × FC_UK = {c_kapasitas:,.1f} smp/jam</strong>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t_col2:
        st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>🚘</span> Rincian Konversi Volume Kendaraan ke smp/jam
</div>""", unsafe_allow_html=True)

        df_volume = pd.DataFrame({
            "Golongan Kendaraan": ["Sepeda Motor (SM)", "Mobil Penumpang (MP)", "Kendaraan Sedang (KS)", "Bus Besar (BB)", "Truk Berat (TB)", "TOTAL ARUS"],
            "Volume (kend/jam)": [f"{in_sm:,}", f"{in_mp:,}", f"{in_ks:,}", f"{in_bb:,}", f"{in_tb:,}", f"{total_kendaraan:,}"],
            "EMP": [f"{emp_sm:.2f}", f"{emp_mp:.2f}", f"{emp_ks:.2f}", f"{emp_bb:.2f}", f"{emp_tb:.2f}", "-"],
            "Arus (smp/jam)": [f"{q_sm:,.1f}", f"{q_mp:,.1f}", f"{q_ks:,.1f}", f"{q_bb:,.1f}", f"{q_tb:,.1f}", f"{q_total:,.1f}"]
        })
        st.dataframe(df_volume, use_container_width=True, hide_index=True)
        st.markdown(f"""<div style="font-family:'JetBrains Mono', monospace; font-size:12px; background:#f1f5f9; padding:10px 12px; border-radius:8px; margin-top:8px; border:1px solid #cbd5e1;">
<strong>Derajat Kejenuhan DS = Q / C = {ds:.3f} <span style="color:{los_info['warna']};">({los_info['status']})</span></strong>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Rekomendasi & Download
    r_col1, r_col2 = st.columns([7, 3])
    with r_col1:
        st.markdown(f"""<div class="content-box" style="border-left:5px solid {los_info['warna']};">
<div class="content-box-title">
<span>📋</span> Evaluasi & Rekomendasi Teknis Manajemen Rekayasa Lalu Lintas
</div>
<p style="font-size:14px; margin-bottom:8px;"><strong>Kondisi:</strong> {los_info['keterangan']}</p>
<div style="background:{los_info['bg_color']}; color:{los_info['text_color']}; padding:12px 14px; border-radius:8px; font-weight:600; font-size:13.5px; margin-bottom:12px;">
💡 <strong>Rekomendasi Ahli Transportasi:</strong><br>{los_info['rekomendasi']}
</div>
</div>""", unsafe_allow_html=True)

    with r_col2:
        st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>📥</span> Ekspor Laporan
</div>""", unsafe_allow_html=True)
        df_export = pd.DataFrame({
            "Waktu": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Tipe_Jalan": [tipe_jalan], "Arus_Q_smp": [round(q_total, 2)],
            "Kapasitas_C_smp": [round(c_kapasitas, 2)], "DS": [round(ds, 3)], "LOS": [los_info["los"]]
        })
        st.download_button(
            label="📄 Unduh Rekap CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"TRANS_METRICS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =================================================================================================
# KONTEN TAB 2: INTEGRASI SMART CCTV DISHUB (REAL-TIME OPENCV + YOLO11 + BYTETRACK)
# =================================================================================================
with tab2:
    st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>📹</span> Pemantauan Video Lalu Lintas Cerdas (Real-Time AI Vision Engine)
</div>
<p style="font-size:13.5px; color:#64748b; margin-top:-8px; margin-bottom:16px;">
Modul ini memproses video CCTV nyata secara frame-by-frame menggunakan model deep learning <strong>Ultralytics YOLO11</strong> 
yang dipadukan dengan algoritma multi-object tracking <strong>ByteTrack</strong>. Kendaraan diidentifikasi ke dalam 
<strong>5 Kelas Standar PKJI 2023</strong> dan dihitung secara live saat melintasi <strong>Virtual Counting Line (ROI)</strong>.
</p>""", unsafe_allow_html=True)

    # 1. PENGATURAN MODEL AI & SUMBER VIDEO
    set_c1, set_c2 = st.columns([5, 7])

    with set_c1:
        st.markdown("<h4 style='font-size:14px; font-weight:700; margin-bottom:6px;'>🧠 Pilihan Model AI Ultralytics</h4>", unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Model Object Detection:",
            options=[
                "YOLO11 Nano (yolo11n.pt) - Ultra Cepat & Real-Time",
                "YOLO11 Small (yolo11s.pt) - Akurasi Tinggi untuk Motor/Objek Kecil",
                "YOLOv8 Nano (yolov8n.pt) - Model Baseline"
            ],
            index=0,
            help="Pilih arsitektur model vision. YOLO11n sangat optimal untuk inferensi CPU berkecepatan tinggi."
        )

        model_map = {
            "YOLO11 Nano (yolo11n.pt) - Ultra Cepat & Real-Time": "yolo11n.pt",
            "YOLO11 Small (yolo11s.pt) - Akurasi Tinggi untuk Motor/Objek Kecil": "yolo11s.pt",
            "YOLOv8 Nano (yolov8n.pt) - Model Baseline": "yolov8n.pt"
        }
        active_model_file = model_map[model_choice]

    with set_c2:
        st.markdown("<h4 style='font-size:14px; font-weight:700; margin-bottom:6px;'>🎬 Pilihan Sumber Video</h4>", unsafe_allow_html=True)
        video_source_type = st.radio(
            "Pilih Sumber Input Video:",
            options=[
                "Video Sampel Lalu Lintas Otomatis",
                "Unggah Video Rekaman CCTV (.mp4 / .avi)",
                "Stream URL CCTV (RTSP / HTTP .m3u8 / YouTube Live / Webcam)"
            ],
            index=0,
            horizontal=True
        )

    # Input Dinamis berdasarkan Pilihan Sumber
    video_path_to_open = None

    if video_source_type == "Unggah Video Rekaman CCTV (.mp4 / .avi)":
        uploaded_file = st.file_uploader("Unggah Berkas Video CCTV:", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path_to_open = tfile.name
            st.success(f"Video terunggah: **{uploaded_file.name}** ({uploaded_file.size/(1024*1024):.1f} MB)")
        else:
            st.info("Silakan unggah berkas video untuk memulai deteksi.")

    elif "Stream URL CCTV" in video_source_type:
        stream_mode = st.radio(
            "Metode Pemilihan Aliran CCTV:",
            options=[
                "🌐 Input URL Bebas (Web CCTV Pemkot / YouTube Live / .m3u8 / RTSP / Webcam)",
                "🏛️ Preset Titik Live CCTV Kota Magelang (40+ Kamera)"
            ],
            index=0,
            horizontal=True
        )

        if "Magelang" in stream_mode:
            magelang_names = list(st.session_state.magelang_channels.keys())
            sel_cam_name = st.selectbox(
                "Pilih Titik Kamera Live CCTV Kota Magelang (Dishub/Kominfo):",
                options=magelang_names,
                index=0,
                help="Pilih salah satu dari 40+ titik CCTV persimpangan dan ruas jalan di Kota Magelang untuk dipantau secara langsung."
            )
            selected_uuid = st.session_state.magelang_channels.get(sel_cam_name, "67338104-0028-4720-9f47-110fd4c0e7a9")
            resolved_stream_url = f"https://dev.cctv.magelangkota.go.id/memfs/{selected_uuid}.m3u8"
            video_path_to_open = resolved_stream_url

            st.markdown(f"""<div style="font-size:12.5px; background:#f0fdf4; color:#166534; padding:8px 12px; border-radius:6px; border:1px solid #bbf7d0; margin-top:4px;">
🟢 <strong>Terhubung ke Titik Kamera:</strong> {sel_cam_name} <br>
<span style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#15803d;">📡 Stream HLS (.m3u8): {resolved_stream_url}</span>
</div>""", unsafe_allow_html=True)

        else:
            col_url, col_preset = st.columns([7, 5])
            with col_preset:
                preset_choice = st.selectbox(
                    "Preset Cepat / Contoh URL:",
                    options=[
                        "Kamera RSJ - Utara Kota Magelang (Web .html)",
                        "YouTube Live CCTV Dishub (Format Siaran Langsung)",
                        "Kamera Webcam Laptop / USB (0)",
                        "Alun-alun Magelang (Gardena HLS .m3u8)",
                        "Simpang GOR Magelang (HLS .m3u8)",
                        "RTSP Stream Lokal (Simulasi)"
                    ]
                )
            with col_url:
                if "Webcam" in preset_choice:
                    default_input = "0"
                elif "RSJ - Utara" in preset_choice:
                    default_input = "https://dev.cctv.magelangkota.go.id/playersite_67338104-0028-4720-9f47-110fd4c0e7a9.html"
                elif "YouTube" in preset_choice:
                    default_input = "https://www.youtube.com/live/UtHqoSIKaqY?si=tR1MrN179Nv8cONH"
                elif "Gardena" in preset_choice:
                    default_input = "https://dev.cctv.magelangkota.go.id/memfs/8fb93fb3-cf14-4d29-8ec5-6ae8d56419df.m3u8"
                elif "Simpang GOR" in preset_choice:
                    default_input = "https://dev.cctv.magelangkota.go.id/memfs/99d847b9-dd05-476a-bf55-b8a06251ec8c.m3u8"
                else:
                    default_input = "rtsp://admin:pass@192.168.1.100:554/live"

                input_stream = st.text_input(
                    "Stream URL, Link Web CCTV, atau Link YouTube Live:",
                    value=default_input,
                    help="Tempelkan link web CCTV pemkot (.html), link YouTube Live CCTV Dishub, link stream langsung (.m3u8/.mp4/rtsp), atau ketik '0' untuk webcam."
                )

            # Resolusi otomatis URL CCTV (Universal 4-Layer Extractor)
            res_url, cam_title, discovered_chs, msg, is_extracted = resolve_cctv_url(input_stream)
            video_path_to_open = res_url

            # Jika ditemukan channel baru dari halaman web, sinkronkan ke daftar
            if discovered_chs:
                st.session_state.magelang_channels.update(discovered_chs)

            if is_extracted and isinstance(res_url, str) and res_url != input_stream.strip():
                if "YouTube" in cam_title or "youtube.com" in str(input_stream).lower() or "youtu.be" in str(input_stream).lower():
                    st.markdown(f"""<div style="font-size:12.5px; background:#fef2f2; color:#991b1b; padding:8px 12px; border-radius:6px; border:1px solid #fecaca; margin-top:4px;">
▶️ <strong>YouTube Live Stream Terdeteksi (via yt-dlp):</strong> Aliran live HLS berhasil diekstrak untuk <strong>{cam_title}</strong>: <br>
<span style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#b91c1c;">📡 {res_url[:120]}...</span>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="font-size:12.5px; background:#eff6ff; color:#1d4ed8; padding:8px 12px; border-radius:6px; border:1px solid #bfdbfe; margin-top:4px;">
⚡ <strong>Smart CCTV Web & Iframe Extractor Berhasil:</strong> Aliran video HLS ditemukan untuk <strong>{cam_title}</strong>: <br>
<span style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#2563eb;">📡 {res_url}</span>
</div>""", unsafe_allow_html=True)
            elif is_extracted and isinstance(res_url, str) and (res_url.endswith(".m3u8") or ".m3u8?" in res_url or res_url.startswith("rtsp://")):
                st.markdown(f"""<div style="font-size:12.5px; background:#f0fdf4; color:#166534; padding:8px 12px; border-radius:6px; border:1px solid #bbf7d0; margin-top:4px;">
🟢 <strong>Direct Stream Terhubung:</strong> Menggunakan aliran video langsung ({res_url[:100]})
</div>""", unsafe_allow_html=True)
            elif str(input_stream).strip() == "0":
                st.info("Kamera Webcam Laptop / USB lokal siap digunakan.")
            elif not is_extracted and any(p in str(input_stream).lower() for p in ["youtube.com", "youtu.be"]):
                st.error(f"❌ Gagal mengekstrak siaran YouTube Live: {msg}")
            elif not is_extracted and str(input_stream).strip().startswith(("http://", "https://")) and str(input_stream).strip() != default_input:
                st.warning(f"⚠️ URL belum terdeteksi sebagai stream langsung: {msg}")

            # Expander Panduan Tips Universal CCTV
            with st.expander("💡 Tips Membuka CCTV Kota Mana Pun di Indonesia"):
                st.markdown("""
                Aplikasi **TRANS-METRICS** mendukung pemantauan CCTV lalu lintas dari kota mana pun di Indonesia (Dishub, Pemkot, Kepolisian / NTMC Korlantas, Jasa Marga) melalui 3 metode mudah:

                1. 🌐 **Link Web CCTV Pemkot / Dishub (.html / Portal CCTV):**
                   * Cukup salin URL halaman web CCTV kota Anda (contoh: portal CCTV Kota Magelang, Bandung, Semarang, Denpasar, Surabaya, Solo, dll).
                   * Engine *Universal Web & Iframe Scraper* akan otomatis membedah kode HTML halaman, menembus tag `<iframe>`, dan mengekstrak aliran stream HLS (`.m3u8`) aslinya.

                2. ▶️ **Link YouTube Live CCTV Dishub:**
                   * Banyak Dinas Perhubungan di Indonesia (seperti Dishub Kota Bandung, Dishub DIY Yogyakarta, Dishub Banyumas, ATCS Denpasar, dll.) menyiarkan CCTV lalu lintas secara langsung melalui YouTube Live.
                   * Cukup tempel link YouTube siaran tersebut (contoh: `https://www.youtube.com/watch?v=...` atau `https://youtu.be/...`), pustaka **yt-dlp** akan mengekstrak siaran stream resolusi terbaik secara otomatis.

                3. 🔍 **Link .m3u8 Langsung (Melalui Fitur Inspect Element Browser):**
                   * Jika portal CCTV kota Anda memiliki proteksi login khusus atau enkripsi JavaScript:
                     1. Buka situs CCTV tersebut di Google Chrome atau Microsoft Edge.
                     2. Tekan tombol **F12** pada keyboard (atau klik kanan -> *Inspect*), lalu pilih tab **Network**.
                     3. Pada kolom pencarian/filter di tab Network, ketik `.m3u8` atau `live`.
                     4. Klik tombol Play pada video di web tersebut, lalu klik kanan pada berkas `.m3u8` yang muncul di daftar Network -> pilih *Copy* -> *Copy link address*.
                     5. Tempelkan link tersebut ke dalam kotak input URL di atas.
                """)

    else: # Video Sampel Otomatis
        sample_path = ensure_sample_video()
        video_path_to_open = sample_path
        st.success(f"✅ Video sampel lalu lintas jalan raya siap diproses: **{os.path.basename(sample_path)}** (Resolusi 960x540 • 335 Frame)")

    st.markdown("---")

    # 2. KONTROL GARIS HITUNG & PARAMETER AI
    st.markdown("<h4 style='font-size:15px; font-weight:700; color:#0f172a; margin-bottom:12px;'>📏 Kontrol Garis Hitung Virtual & Parameter AI</h4>", unsafe_allow_html=True)
    c_s1, c_s2 = st.columns(2)

    with c_s1:
        line_pos_pct = st.slider(
            "Posisi Garis Hitung Vertikal (% Tinggi Frame):",
            min_value=20,
            max_value=85,
            value=65,
            step=1,
            format="%d%%",
            help="Menentukan elevasi garis virtual melintang jalan tempat kendaraan dicacah (ROI)."
        )

    with c_s2:
        conf_thresh = st.slider(
            "Confidence Threshold AI:",
            min_value=0.10,
            max_value=0.80,
            value=0.20,
            step=0.05,
            format="%.2f",
            help="Ambang batas keyakinan deteksi objek. Default 0.20 optimal untuk malam hari, aspal basah, dan silau lampu."
        )

    enable_clahe = st.checkbox(
        "🌙 Mode Peningkatan Kontras CCTV Malam / Hujan (CLAHE Auto-Enhance)",
        value=True,
        help="Menerapkan Contrast Limited Adaptive Histogram Equalization pada kanal L (LAB) sebelum inferensi YOLO11 untuk memperjelas siluet kendaraan pada rekaman minim cahaya/silau malam hari."
    )

    # Tombol Start, Stop, & Reset
    col_act1, col_act2, col_act3 = st.columns([4, 4, 4])
    with col_act1:
        start_tracking = st.button("▶️ Mulai Deteksi & Pencacahan YOLO11", type="primary", use_container_width=True)
    with col_act2:
        stop_tracking = st.button("⏹️ Hentikan & Simpan Hasil", use_container_width=True)
    with col_act3:
        reset_counter = st.button("🔄 Reset Hitungan AI", use_container_width=True)

    if reset_counter:
        st.session_state.cctv_counts = {"SM": 0, "MP": 0, "KS": 0, "BB": 0, "TB": 0}
        st.session_state.cctv_dir_counts = {"Arah 1": 0, "Arah 2": 0}
        st.session_state.cctv_elapsed_sec = 0.0
        st.session_state.cctv_frame_count = 0
        st.session_state.last_annotated_frame = None
        st.session_state.tracking_active = False
        st.toast("Counter kendaraan dan histori video berhasil di-reset!", icon="🔄")
        st.rerun()

    if stop_tracking:
        st.session_state.tracking_active = False

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. METODE KONVERSI ARUS PER JAM (Q)
    st.markdown("""<div class="content-box" style="padding:16px 20px; margin-bottom:18px;">
<div style="font-size:14px; font-weight:700; color:#0f172a; margin-bottom:8px;">
⏱️ Metode Konversi Arus Per Jam (Q) Berdasarkan Durasi Video
</div>""", unsafe_allow_html=True)
    conv_method = st.radio(
        "Pilih Dasar Proyeksi Jam:",
        options=[
            "Ekstrapolasi Durasi Video Aktual (3600 / durasi_video_detik) [Direkomendasikan untuk Video Pendek]",
            "Pengali Standar Interval 15 Menit (Count x 4)"
        ],
        index=0,
        horizontal=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. DISPLAY LIVE REAL-TIME (VIDEO DI KIRI, TABEL LIVE DI KANAN)
    disp_col1, disp_col2 = st.columns([7, 5])

    with disp_col1:
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        download_btn_placeholder = st.empty()

    with disp_col2:
        table_placeholder = st.empty()
        dir_metric_placeholder = st.empty()
        apply_placeholder = st.empty()

    # Fungsi untuk memperbarui tampilan tabel live di kolom kanan
    def render_live_table(counts_dict, placeholder, elapsed_sec=0.0, frame_count=0, video_fps=25.0):
        sm = counts_dict.get("SM", 0)
        mp = counts_dict.get("MP", 0)
        ks = counts_dict.get("KS", 0)
        bb = counts_dict.get("BB", 0)
        tb = counts_dict.get("TB", 0)
        total_counted = sm + mp + ks + bb + tb

        # Hitung Faktor Pengali Sesuai Metode Terpilih
        if "Ekstrapolasi" in conv_method:
            eff_sec = max(1.0, elapsed_sec) if elapsed_sec > 0 else 11.2 # default durasi sampel
            multiplier = 3600.0 / eff_sec
            subtext_method = f"Ekstrapolasi Waktu Riil: {eff_sec:.1f} detik ({frame_count} frame @ {video_fps:.1f} FPS) • Pengali: <strong>{multiplier:.2f}x</strong>"
        else:
            multiplier = 4.0
            subtext_method = "Pengali Standar Interval 15 Menit (Peak Hour Factor PHF = 4x)"

        proj_sm = int(round(sm * multiplier))
        proj_mp = int(round(mp * multiplier))
        proj_ks = int(round(ks * multiplier))
        proj_bb = int(round(bb * multiplier))
        proj_tb = int(round(tb * multiplier))
        proj_total = int(round(total_counted * multiplier))

        html_table = f"""<div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px; padding:18px; box-shadow:0 2px 4px rgba(0,0,0,0.04);">
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #cbd5e1; padding-bottom:8px; margin-bottom:12px;">
<span style="font-size:14.5px; font-weight:800; color:#0f172a;">📊 Hasil Pencacahan Otomatis AI (Live)</span>
<span style="font-size:11px; background:#dcfce7; color:#15803d; font-weight:700; padding:3px 8px; border-radius:99px;">ByteTrack Active</span>
</div>
<table style="width:100%; font-size:13px; border-collapse:collapse; margin-bottom:12px;">
<tr style="border-bottom:2px solid #e2e8f0; color:#64748b;">
<th style="text-align:left; padding:6px 0;">Klasifikasi PKJI 2023</th>
<th style="text-align:right; padding:6px 0;">Hitungan Riil</th>
<th style="text-align:right; padding:6px 0;">Proyeksi / Jam</th>
</tr>
<tr style="border-bottom:1px solid #f1f5f9;">
<td style="padding:6px 0; font-weight:600; color:#0284c7;">🛵 Sepeda Motor (SM)</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-weight:700; font-size:14px;">{sm}</td>
<td style="text-align:right; font-family:'JetBrains Mono'; color:#0284c7; font-weight:700;">{proj_sm:,}</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9;">
<td style="padding:6px 0; font-weight:600; color:#16a34a;">🚙 Mobil Penumpang (MP)</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-weight:700; font-size:14px;">{mp}</td>
<td style="text-align:right; font-family:'JetBrains Mono'; color:#16a34a; font-weight:700;">{proj_mp:,}</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9;">
<td style="padding:6px 0; font-weight:600; color:#0d9488;">🚐 Kendaraan Sedang (KS)</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-weight:700; font-size:14px;">{ks}</td>
<td style="text-align:right; font-family:'JetBrains Mono'; color:#0d9488; font-weight:700;">{proj_ks:,}</td>
</tr>
<tr style="border-bottom:1px solid #f1f5f9;">
<td style="padding:6px 0; font-weight:600; color:#d97706;">🚌 Bus Besar (BB)</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-weight:700; font-size:14px;">{bb}</td>
<td style="text-align:right; font-family:'JetBrains Mono'; color:#d97706; font-weight:700;">{proj_bb:,}</td>
</tr>
<tr style="border-bottom:2px solid #e2e8f0;">
<td style="padding:6px 0; font-weight:600; color:#dc2626;">🚛 Truk Berat (TB)</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-weight:700; font-size:14px;">{tb}</td>
<td style="text-align:right; font-family:'JetBrains Mono'; color:#dc2626; font-weight:700;">{proj_tb:,}</td>
</tr>
<tr style="font-weight:800; color:#0f172a; font-size:13.5px;">
<td style="padding:8px 0;">TOTAL VOLUME</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-size:15px; color:#0f172a;">{total_counted}</td>
<td style="text-align:right; font-family:'JetBrains Mono'; font-size:15px; color:#0284c7;">{proj_total:,}</td>
</tr>
</table>
<div style="font-size:11.5px; color:#64748b; line-height:1.4;">
ℹ️ <strong>Dasar Konversi:</strong> {subtext_method}
</div>
</div>"""
        placeholder.markdown(html_table, unsafe_allow_html=True)
        return {
            "proj_sm": proj_sm, "proj_mp": proj_mp, "proj_ks": proj_ks,
            "proj_bb": proj_bb, "proj_tb": proj_tb, "proj_total": proj_total
        }

    # Render Tabel Awal
    last_proj = render_live_table(
        st.session_state.cctv_counts,
        table_placeholder,
        elapsed_sec=st.session_state.cctv_elapsed_sec,
        frame_count=st.session_state.cctv_frame_count,
        video_fps=st.session_state.cctv_video_fps
    )

    # Render Metrik Pemisahan Arah (Directional Split)
    dir_c1 = st.session_state.cctv_dir_counts.get("Arah 1", 0)
    dir_c2 = st.session_state.cctv_dir_counts.get("Arah 2", 0)
    rec_pa, p_dir1, p_dir2 = hitung_rekomendasi_pa(dir_c1, dir_c2)

    def render_dir_metric(placeholder, c1, c2, p1, p2, pa_rec):
        html_dir = f"""<div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:10px; padding:12px 14px; margin-top:10px;">
<div style="font-size:11.5px; font-weight:700; color:#475569; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;">
🧭 Pemisahan Arah Otomatis (Directional Split / PA)
</div>
<div style="font-size:13.5px; font-weight:800; color:#0f172a; margin-bottom:4px;">
⬇️ Arah 1: {c1} kend ({p1:.1f}%) &nbsp;|&nbsp; ⬆️ Arah 2: {c2} kend ({p2:.1f}%)
</div>
<div style="font-size:12px; color:#0284c7; font-weight:600;">
👉 Rekomendasi Parameter PA PKJI: <span style="background:#0284c7; color:#ffffff; padding:2px 8px; border-radius:4px; font-weight:800;">{pa_rec}</span>
</div>
</div>"""
        placeholder.markdown(html_dir, unsafe_allow_html=True)

    render_dir_metric(dir_metric_placeholder, dir_c1, dir_c2, p_dir1, p_dir2, rec_pa)

    # TAMPILAN STANDBY / PREVIEW FRAME (AGAR TIDAK KOSONG PUTIH)
    if not st.session_state.tracking_active:
        if st.session_state.last_annotated_frame is not None:
            # Tampilkan frame terakhir yang telah diproses
            video_placeholder.image(st.session_state.last_annotated_frame, channels="RGB", use_container_width=True)
            info_placeholder.markdown(f"""<div style="font-family:'JetBrains Mono', monospace; font-size:12px; background:#0f172a; color:#38bdf8; padding:8px 12px; border-radius:6px; margin-top:6px;">
● FRAME CAPTURE TERAKHIR | Terproses: {st.session_state.cctv_frame_count} Frame ({st.session_state.cctv_elapsed_sec:.1f}s) | Status: Tersimpan & Standby
</div>""", unsafe_allow_html=True)
        else:
            # Jika baru dibuka pertama kali, ambil thumbnail preview frame pertama
            if video_path_to_open is not None:
                preview_img = get_video_preview(video_path_to_open, line_pos_pct, enable_clahe=enable_clahe)
                if preview_img is not None:
                    st.session_state.last_annotated_frame = preview_img
                    video_placeholder.image(preview_img, channels="RGB", use_container_width=True)
                    info_placeholder.markdown(f"""<div style="font-family:'JetBrains Mono', monospace; font-size:12px; background:#0f172a; color:#94a3b8; padding:8px 12px; border-radius:6px; margin-top:6px;">
● THUMBNAIL PREVIEW | Model: {active_model_file} | Garis Hitung: {line_pos_pct}% | Klik 'Mulai Deteksi' untuk memproses
</div>""", unsafe_allow_html=True)

    # TOMBOL UNDUH BUKTI CAPTURE SURVEI CCTV
    if st.session_state.last_annotated_frame is not None:
        frame_bgr_dl = cv2.cvtColor(st.session_state.last_annotated_frame, cv2.COLOR_RGB2BGR)
        ret_enc, img_bytes = cv2.imencode('.jpg', frame_bgr_dl, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ret_enc:
            download_btn_placeholder.download_button(
                label="📸 Unduh Bukti Capture Deteksi AI (.jpg)",
                data=img_bytes.tobytes(),
                file_name=f"BUKTI_SURVEI_CCTV_YOLO11_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

    # TOMBOL APPLY KE TAB 1
    with apply_placeholder.container():
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Terapkan Hasil Hitungan CCTV ke Tab 1", type="primary", use_container_width=True):
            counts = st.session_state.cctv_counts
            total_current = sum(counts.values())

            if total_current == 0:
                # Default baseline jika belum menjalankan video
                st.session_state.vol_sm = 2048
                st.session_state.vol_mp = 840
                st.session_state.vol_ks = 128
                st.session_state.vol_bb = 56
                st.session_state.vol_tb = 36
                st.session_state.pemisahan_arah = "50-50"
                total_applied = 3108
            else:
                # Terapkan hasil proyeksi per jam yang akurat
                st.session_state.vol_sm = max(10, last_proj["proj_sm"])
                st.session_state.vol_mp = max(10, last_proj["proj_mp"])
                st.session_state.vol_ks = max(5, last_proj["proj_ks"])
                st.session_state.vol_bb = max(2, last_proj["proj_bb"])
                st.session_state.vol_tb = max(2, last_proj["proj_tb"])
                st.session_state.pemisahan_arah = rec_pa
                total_applied = last_proj["proj_total"]

            st.session_state.cctv_applied = True
            st.session_state.cctv_applied_total = total_applied
            st.session_state.cctv_applied_pa = rec_pa
            st.toast(f"✅ Berhasil sinkronisasi {total_applied:,} kend/jam & PA {rec_pa} ke Tab 1!", icon="🚦")
            st.rerun()

    # ---------------------------------------------------------------------------------------------
    # 5. EKSEKUSI LOOP DETEKSI & TRACKING JIKA TOMBOL START DITEKAN
    # ---------------------------------------------------------------------------------------------
    if start_tracking and video_path_to_open is not None:
        st.session_state.tracking_active = True
        
        # Load Model
        with st.spinner(f"Memuat model {active_model_file}..."):
            model = load_yolo_model(active_model_file)

        # Buka Video menggunakan SmartVideoSource (Mendukung File, Webcam, HLS & Fallback TS)
        cap = SmartVideoSource(video_path_to_open)
        if not cap.isOpened():
            st.error(f"Gagal membuka sumber video: {video_path_to_open}. Silakan periksa koneksi internet atau format URL.")
            st.session_state.tracking_active = False
        else:
            frame_w = cap.frame_w
            frame_h = cap.frame_h
            video_fps = cap.fps
            if video_fps is None or video_fps <= 0 or np.isnan(video_fps):
                video_fps = 25.0
            if frame_w == 0 or frame_h == 0:
                frame_w, frame_h = 800, 480

            line_y = int(frame_h * (line_pos_pct / 100.0))
            counted_ids = set()
            track_history = {} # track_id -> {"first_y": int, "last_y": int, "frames_seen": int}
            green_flash_frames = 0
            last_crossed_label = ""

            # Salin count yang sudah ada
            current_counts = dict(st.session_state.cctv_counts)
            dir_counts = dict(st.session_state.cctv_dir_counts)

            prev_time = time.time()
            frame_idx = 0

            # Running loop
            while cap.isOpened() and st.session_state.tracking_active:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                elapsed_video_sec = frame_idx / float(video_fps)

                # Preprocessing Peningkatan Kontras Malam / Hujan (CLAHE)
                if enable_clahe:
                    detection_frame = apply_clahe_enhancement(frame, clip_limit=2.5, tile_grid_size=(8, 8))
                else:
                    detection_frame = frame

                # Jalankan Tracking YOLO11 + ByteTrack
                results = model.track(
                    detection_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[1, 2, 3, 5, 7],
                    conf=conf_thresh,
                    imgsz=640,
                    verbose=False
                )

                annotated_frame = detection_frame.copy() if enable_clahe else frame.copy()

                # Cek jika ada objek terdeteksi dengan ID pelacak
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.int().cpu().tolist()
                    confs = results[0].boxes.conf.cpu().tolist()

                    for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                        x1, y1, x2, y2 = map(int, box)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Update histori koordinat pelacak (Anti-Miss Crossing)
                        if track_id not in track_history:
                            track_history[track_id] = {
                                "first_y": cy,
                                "last_y": cy,
                                "frames_seen": 1
                            }
                        else:
                            track_history[track_id]["last_y"] = cy
                            track_history[track_id]["frames_seen"] += 1

                        first_y = track_history[track_id]["first_y"]
                        frames_seen = track_history[track_id]["frames_seen"]

                        box_area = (x2 - x1) * (y2 - y1)
                        area_ratio = box_area / float(frame_w * frame_h)

                        # Pemetaan Kelas COCO ke 5 Kelas PKJI 2023 (Class 1 & 3 dipetakan ke SM)
                        if cls_id in (1, 3): # 1: bicycle, 3: motorcycle -> Keduanya Sepeda Motor (SM)
                            pkji_code = "SM"
                            color_bgr = (230, 140, 2) # Blue
                        elif cls_id == 2: # car -> Mobil Penumpang (MP)
                            pkji_code = "MP"
                            color_bgr = (34, 197, 94) # Green
                        elif cls_id == 5: # bus -> Bus Besar (BB)
                            pkji_code = "BB"
                            color_bgr = (11, 158, 245) # Orange
                        elif cls_id == 7: # truck
                            if area_ratio < 0.045:
                                pkji_code = "KS" # Kendaraan Sedang
                                color_bgr = (129, 185, 16) # Toska
                            else:
                                pkji_code = "TB" # Truk Berat
                                color_bgr = (68, 68, 239) # Red
                        else:
                            continue

                        # Gambar Bounding Box & Label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
                        cv2.circle(annotated_frame, (cx, cy), 4, color_bgr, -1)

                        label_text = f"{pkji_code} #{track_id} [{conf:.2f}]"
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color_bgr, -1)
                        cv2.putText(annotated_frame, label_text, (x1 + 3, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                        # Logika Pelintasan Garis Hitung Virtual (Anti-Miss Crossing)
                        if track_id not in counted_ids:
                            crossed = False
                            direction_detected = None

                            # Kriteria 1: Bergerak Turun / Mendekati Kamera (first_y < line_y dan cy >= line_y)
                            if first_y < line_y and cy >= line_y:
                                crossed = True
                                direction_detected = "Arah 1"

                            # Kriteria 2: Bergerak Naik / Menjauhi Kamera (first_y > line_y dan cy <= line_y)
                            elif first_y > line_y and cy <= line_y:
                                crossed = True
                                direction_detected = "Arah 2"

                            # Kriteria 3: Terdeteksi di dalam sabuk garis (abs(cy - line_y) <= 18 px) setelah min 2 frame
                            elif abs(cy - line_y) <= 18 and frames_seen >= 2:
                                crossed = True
                                if cy > first_y:
                                    direction_detected = "Arah 1"
                                elif cy < first_y:
                                    direction_detected = "Arah 2"
                                else:
                                    direction_detected = "Arah 1" if cy >= line_y else "Arah 2"

                            if crossed:
                                counted_ids.add(track_id)
                                current_counts[pkji_code] += 1
                                if direction_detected in dir_counts:
                                    dir_counts[direction_detected] += 1
                                else:
                                    dir_counts["Arah 1"] += 1
                                green_flash_frames = 10
                                last_crossed_label = f"+1 {pkji_code} (#{track_id})"

                # Gambar Garis Hitung Virtual (ROI) & Efek Flash Hijau Neon
                if green_flash_frames > 0:
                    line_color = (0, 255, 0) # HIJAU NEON TEBAL
                    line_thick = 4
                    cv2.line(annotated_frame, (0, line_y), (frame_w, line_y), line_color, line_thick)
                    line_banner = f"--- VIRTUAL COUNTING LINE (ROI {line_pos_pct}%) [COUNT: {last_crossed_label}] ---"
                    cv2.putText(annotated_frame, line_banner, (20, max(28, line_y - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)

                    # Pop-up Badge Visual Notifikasi Melintas
                    if last_crossed_label:
                        badge_text = f"CROSSED: {last_crossed_label}"
                        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        badge_x = max(20, min(frame_w - bw - 40, int(frame_w * 0.38)))
                        badge_y = max(bh + 12, line_y - 18)
                        cv2.rectangle(annotated_frame, (badge_x - 10, badge_y - bh - 8), (badge_x + bw + 10, badge_y + 8), (0, 180, 0), -1)
                        cv2.rectangle(annotated_frame, (badge_x - 10, badge_y - bh - 8), (badge_x + bw + 10, badge_y + 8), (255, 255, 255), 2)
                        cv2.putText(annotated_frame, badge_text, (badge_x, badge_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    green_flash_frames -= 1
                else:
                    line_color = (0, 0, 255) # MERAH DEFAULT
                    line_thick = 2
                    cv2.line(annotated_frame, (0, line_y), (frame_w, line_y), line_color, line_thick)
                    cv2.putText(annotated_frame, f"--- VIRTUAL COUNTING LINE (ROI {line_pos_pct}%) ---", (20, max(28, line_y - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)

                # Hitung FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
                prev_time = curr_time

                # Header HUD Overlay pada video
                cv2.rectangle(annotated_frame, (0, 0), (frame_w, 36), (15, 23, 42), -1)
                clahe_status = "CLAHE: ON" if enable_clahe else "CLAHE: OFF"
                hud_text = f"MODEL: {active_model_file.upper()} | {clahe_status} | FPS: {fps:.1f} | TRACKS: {len(results[0].boxes)} | TOTAL: {sum(current_counts.values())}"
                cv2.putText(annotated_frame, hud_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 2)

                # Konversi BGR ke RGB
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update live info & tabel di samping
                info_placeholder.markdown(f"""<div style="font-family:'JetBrains Mono', monospace; font-size:12px; background:#0f172a; color:#38bdf8; padding:8px 12px; border-radius:6px; margin-top:6px;">
● LIVE STREAMING | Frame: {frame_idx} ({elapsed_video_sec:.1f}s) | FPS: {fps:.1f} | Resolusi: {frame_w}x{frame_h} | Model: {active_model_file}
</div>""", unsafe_allow_html=True)

                last_proj = render_live_table(
                    current_counts,
                    table_placeholder,
                    elapsed_sec=elapsed_video_sec,
                    frame_count=frame_idx,
                    video_fps=video_fps
                )

                # Update live Directional Split
                r_pa, p1, p2 = hitung_rekomendasi_pa(dir_counts["Arah 1"], dir_counts["Arah 2"])
                render_dir_metric(dir_metric_placeholder, dir_counts["Arah 1"], dir_counts["Arah 2"], p1, p2, r_pa)

                # Simpan ke session state berkala
                st.session_state.cctv_counts = current_counts
                st.session_state.cctv_dir_counts = dir_counts
                st.session_state.cctv_elapsed_sec = elapsed_video_sec
                st.session_state.cctv_frame_count = frame_idx
                st.session_state.cctv_video_fps = video_fps
                st.session_state.last_annotated_frame = frame_rgb

            cap.release()
            st.session_state.tracking_active = False
            st.success("✅ Pemrosesan video selesai! Frame dan seluruh hasil hitungan tersimpan secara permanen.")
            st.rerun()

    # 6. ROADMAP ARSITEKTUR COMPUTER VISION & DATA PIPELINE
    st.markdown("""<div class="content-box">
<div class="content-box-title">
<span>🗺️</span> Roadmap Arsitektur AI & Computer Vision untuk Smart Traffic Dishub
</div>
<p style="font-size:13.5px; color:#64748b; margin-top:-8px; margin-bottom:20px;">
Arsitektur terintegrasi end-to-end yang menjembatani streaming video CCTV mentah menjadi angka kinerja jalan (DS dan LOS) berstandar nasional PKJI 2023.
</p>""", unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""<div style="border:1px solid #e2e8f0; border-top:4px solid #0284c7; border-radius:10px; padding:16px; background:#ffffff; height:100%;">
<div style="font-size:24px; margin-bottom:8px;">👁️</div>
<h4 style="margin:0 0 8px 0; font-size:15px; color:#0f172a; font-weight:700;">1. YOLO11 Engine</h4>
<div style="font-size:12px; color:#0284c7; font-weight:700; margin-bottom:8px;">Deep Learning Detection</div>
<p style="font-size:12px; color:#64748b; line-height:1.5; margin:0;">
Generasi terbaru Ultralytics YOLO11 (yolo11n / yolo11s) dengan arsitektur C3k2 & SPPF mutakhir. Inferensi sangat cepat (&lt;35 ms pada CPU) untuk mengenali kendaraan perkotaan.
</p>
</div>""", unsafe_allow_html=True)

    with p2:
        st.markdown("""<div style="border:1px solid #e2e8f0; border-top:4px solid #10b981; border-radius:10px; padding:16px; background:#ffffff; height:100%;">
<div style="font-size:24px; margin-bottom:8px;">🔄</div>
<h4 style="margin:0 0 8px 0; font-size:15px; color:#0f172a; font-weight:700;">2. ByteTrack MOT</h4>
<div style="font-size:12px; color:#10b981; font-weight:700; margin-bottom:8px;">Multi-Object Tracking</div>
<p style="font-size:12px; color:#64748b; line-height:1.5; margin:0;">
Pelacak objek berbasis Kalman Filter dan Hungarian Association. Menjaga ID unik setiap kendaraan meskipun terjadi bayangan atau oklusi parsial oleh truk atau bus besar.
</p>
</div>""", unsafe_allow_html=True)

    with p3:
        st.markdown("""<div style="border:1px solid #e2e8f0; border-top:4px solid #f59e0b; border-radius:10px; padding:16px; background:#ffffff; height:100%;">
<div style="font-size:24px; margin-bottom:8px;">📏</div>
<h4 style="margin:0 0 8px 0; font-size:15px; color:#0f172a; font-weight:700;">3. Virtual Tripwire</h4>
<div style="font-size:12px; color:#f59e0b; font-weight:700; margin-bottom:8px;">Directional Counting Line</div>
<p style="font-size:12px; color:#64748b; line-height:1.5; margin:0;">
Garis hitung virtual dua titik melintang ruas jalan. Mendeteksi penyeberangan garis dan arah pergerakan (Arah 1 vs Arah 2) untuk kalibrasi otomatis faktor Pemisahan Arah (PA).
</p>
</div>""", unsafe_allow_html=True)

    with p4:
        st.markdown("""<div style="border:1px solid #e2e8f0; border-top:4px solid #8b5cf6; border-radius:10px; padding:16px; background:#ffffff; height:100%;">
<div style="font-size:24px; margin-bottom:8px;">📈</div>
<h4 style="margin:0 0 8px 0; font-size:15px; color:#8b5cf6; font-weight:700;">4. Agregasi PHF & Q</h4>
<div style="font-size:12px; color:#8b5cf6; font-weight:700; margin-bottom:8px;">Arus Ekstrapolasi Jam</div>
<p style="font-size:12px; color:#64748b; line-height:1.5; margin:0;">
Mengonversi hitungan durasi video aktual menjadi volume arus kendaraan per jam (kend/jam) menggunakan formula ekstrapolasi waktu riil maupun interval standar 15 menit.
</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<h4 style="font-size:15px; color:#0f172a; font-weight:700; margin-bottom:12px;">
🔄 Alur Pipeline Data Real-Time: Dari CCTV Menuju Keputusan Rekayasa Lalu Lintas
</h4>
<div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:10px; padding:18px; font-family:'JetBrains Mono', monospace; font-size:12.5px; color:#1e293b; overflow-x:auto;">
[Kamera CCTV Dishub / Video File] 
     │
     ▼
[Frame Preprocessing & Scaling (OpenCV)] 
     │
     ▼
[YOLO11 Object Detection] ──► Klasifikasi: SM, MP, KS, BB, TB
     │
     ▼
[ByteTrack Multi-Object Tracker] ──► Pemberian ID Unik & Trajektori Koordinat (X,Y)
     │
     ▼
[Virtual Counting Line Crossing] ──► Trigger Counter saat Garis Terlintasi & Deteksi Arah (PA)
     │
     ▼
[Time-Window Aggregator & Ekstrapolasi] ──► Proyeksi Q (kend/jam)
     │
     ▼
[Konversi Otomatis EMP PKJI 2023] ──► Arus Total Q (smp/jam)
     │
     ▼
[Engine Evaluasi Kinerja (Tab 1)] ──► Derajat Kejenuhan (DS) & Level of Service (LOS)
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 9. FOOTER APLIKASI
# -------------------------------------------------------------------------------------------------
st.markdown("""<div style="text-align:center; padding:20px 0; color:#94a3b8; font-size:12.5px; border-top:1px solid #e2e8f0; margin-top:30px;">
<strong>TRANS-METRICS</strong> • Kalkulator Kinerja Ruas Jalan PKJI 2023 & Smart CCTV Monitor YOLO11<br>
Dikembangkan untuk Keperluan Akademik Tugas Besar Rekayasa Transportasi & Aplikasi Komputer Teknik Sipil • Pedoman Kapasitas Jalan Indonesia 2023
</div>""", unsafe_allow_html=True)
