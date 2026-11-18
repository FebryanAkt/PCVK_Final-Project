import streamlit as st
import cv2
import mediapipe as mp
import sys
from config import (
    APP_TITLE, APP_SUBTITLE, FACE_LANDMARKER_PATH,
    HAND_LANDMARKER_PATH, PDF_JURNAL_PATH, JOURNAL_TITLE, JOURNAL_AUTHORS
)

def render_home_page():
    # Hero Banner
    st.markdown(f"""
    <div class="vl-hero">
        <span class="vl-badge vl-badge-info" style="margin-bottom: 0.6rem;">Final Project Pengolahan Citra & Visi Komputer</span>
        <div class="vl-hero-title">{APP_TITLE}</div>
        <div class="vl-hero-desc">
            Sistem klasifikasi bentuk wajah dan antarmuka visi komputer yang disesuaikan secara langsung dengan metodologi jurnal ilmiah 
            <strong>"{JOURNAL_TITLE}"</strong> karya {JOURNAL_AUTHORS}.
        </div>
        <div style="margin-top: 1.2rem;">
            <span class="vl-badge vl-badge-neutral">13 Facial Landmarks</span>
            <span class="vl-badge vl-badge-neutral">14 PCA Features</span>
            <span class="vl-badge vl-badge-neutral">5 Face Shapes</span>
            <span class="vl-badge vl-badge-neutral">Eyeglass Recommendation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Grid
    st.markdown('<div class="vl-section-header">Status Lingkungan Komputasi</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">Python Runtime</div>
            <div class="vl-stat-val">{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Python Standard Environment</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">OpenCV Engine</div>
            <div class="vl-stat-val">{cv2.__version__}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Image Processing Library</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">MediaPipe Framework</div>
            <div class="vl-stat-val">{mp.__version__}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">478 3D Face Landmarker</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">Streamlit Core</div>
            <div class="vl-stat-val">{st.__version__}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Web UI Framework</div>
        </div>
        """, unsafe_allow_html=True)

    # Verifikasi File Jurnal dan Model
    st.markdown('<div class="vl-section-header">Kesiapan Berkas Jurnal & Model</div>', unsafe_allow_html=True)
    j_col1, j_col2 = st.columns(2)

    with j_col1:
        if PDF_JURNAL_PATH.exists():
            size_mb = round(PDF_JURNAL_PATH.stat().st_size / (1024*1024), 2)
            st.markdown(f"""
            <div class="vl-card" style="border-left: 4px solid #2563EB;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">Berkas Jurnal Referensi Terdeteksi</div>
                    <span class="vl-badge vl-badge-success">Tersedia</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.3rem;">
                    Path: <code>{PDF_JURNAL_PATH.name}</code> ({size_mb} MB)
                </div>
                <div style="font-size: 0.8rem; color: #334155; line-height: 1.4;">
                    Metode: Ekstraksi 13 titik landmark untuk menghitung 7 jarak (D1-D7), 10 rasio (R1-R10), dan 3 sudut (A1-A3) dengan seleksi 14 fitur PCA terbaik.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="vl-card" style="border-left: 4px solid #DC2626;">
                <div style="font-weight: 700; color: #0F172A;">Berkas Jurnal Tidak Ditemukan</div>
                <div style="font-size: 0.8rem; color: #DC2626;">Path: {PDF_JURNAL_PATH}</div>
            </div>
            """, unsafe_allow_html=True)

    with j_col2:
        if FACE_LANDMARKER_PATH.exists():
            size_mb = round(FACE_LANDMARKER_PATH.stat().st_size / (1024*1024), 2)
            st.markdown(f"""
            <div class="vl-card" style="border-left: 4px solid #16A34A;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">Model Face Landmarker</div>
                    <span class="vl-badge vl-badge-success">Tersedia</span>
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.3rem;">
                    File: <code>{FACE_LANDMARKER_PATH.name}</code> ({size_mb} MB)
                </div>
                <div style="font-size: 0.8rem; color: #334155; line-height: 1.4;">
                    Model MediaPipe Tasks berpresisi tinggi dengan 478 koordinat 3D, mencakup area dahi, garis rambut, pelipis, dan rahang yang dipetakan ke 13 titik Dlib jurnal.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="vl-card" style="border-left: 4px solid #DC2626;">
                <div style="font-weight: 700; color: #0F172A;">Model Face Landmarker Tidak Ditemukan</div>
                <div style="font-size: 0.8rem; color: #DC2626;">Path: {FACE_LANDMARKER_PATH}</div>
            </div>
            """, unsafe_allow_html=True)

    # Ringkasan Metodologi Jurnal
    st.markdown('<div class="vl-section-header">Ringkasan Metodologi Jurnal Ilmiah (Mehta & Mahmoud)</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="vl-card">
            <span class="vl-badge vl-badge-info" style="margin-bottom: 0.5rem;">Tahap 1: Landmarking</span>
            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.3rem;">13 Titik Anatomis Wajah</div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
                Mengambil titik koordinat dahi kiri/kanan, garis rambut teratas, tulang pipi (telinga kiri/kanan), sudut rahang (gonion), dan titik-titik pembentuk dagu.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="vl-card">
            <span class="vl-badge vl-badge-info" style="margin-bottom: 0.5rem;">Tahap 2: Ekstraksi Geometri</span>
            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.3rem;">7 Jarak, 10 Rasio, 3 Sudut</div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
                Menghitung jarak Euclidean antar titik kunci, normalisasi jarak terhadap total keliling, kalkulasi 10 rasio proporsi wajah, dan sudut kemiringan rahang.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="vl-card">
            <span class="vl-badge vl-badge-info" style="margin-bottom: 0.5rem;">Tahap 3: Klasifikasi & Rekomendasi</span>
            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.3rem;">14 Fitur PCA & 5 Bentuk Wajah</div>
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">
                Menentukan bentuk wajah (Heart, Oblong, Oval, Round, Square) menggunakan 14 fitur teratas hasil seleksi PCA dan memberikan rekomendasi kacamata serta gaya rambut.
            </div>
        </div>
        """, unsafe_allow_html=True)
