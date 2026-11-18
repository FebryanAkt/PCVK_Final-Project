import streamlit as st
import cv2
import mediapipe as mp
import sys
from config import FACE_LANDMARKER_PATH, JOURNAL_TITLE, JOURNAL_AUTHORS, DATASET_SAMPLES_COUNT

def render_home_page():
    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.3rem;">
            Ringkasan Metodologi Jurnal Ilmiah
        </h2>
        <div style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.55;">
            Sistem klasifikasi bentuk wajah ini mengimplementasikan metodologi riset dari jurnal 
            <strong>"{}"</strong> oleh <em>{}</em>.
        </div>
    </div>
    """.format(JOURNAL_TITLE, JOURNAL_AUTHORS), unsafe_allow_html=True)

    # ── 3 Tahapan Utama Metodologi ─────────────────────────────
    st.markdown('<div class="section-title">3 Tahap Alur Klasifikasi Sesuai Jurnal</div>', unsafe_allow_html=True)
    
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="clean-card" style="height: 100%;">
            <div style="font-size: 0.72rem; font-weight: 700; color: var(--accent); text-transform: uppercase; margin-bottom: 0.35rem;">
                Tahap 01 &bull; Landmarking
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem;">
                13 Titik Anatomis Kunci
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.6;">
                Mengekstrak koordinat dahi terluar, puncak garis rambut, tulang pipi (telinga kiri-kanan), 
                sudut rahang (gonion), dan kontur dagu menggunakan pemetaan MediaPipe 478 3D ke 81 Dlib landmarks.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="clean-card" style="height: 100%;">
            <div style="font-size: 0.72rem; font-weight: 700; color: var(--accent); text-transform: uppercase; margin-bottom: 0.35rem;">
                Tahap 02 &bull; Geometri
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem;">
                7 Jarak, 10 Rasio, 3 Sudut
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.6;">
                Menghitung jarak Euclidean ($D_1-D_7$), normalisasi jarak terhadap total keliling ($N_1-N_7$), 
                10 rasio proporsi wajah ($R_1-R_{10}$), serta 3 sudut kemiringan rahang ($A_1-A_3$).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="clean-card" style="height: 100%;">
            <div style="font-size: 0.72rem; font-weight: 700; color: var(--accent); text-transform: uppercase; margin-bottom: 0.35rem;">
                Tahap 03 &bull; Klasifikasi
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem;">
                14 Fitur PCA &bull; 5 Kelas
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.6;">
                Memilih 14 fitur paling diskriminatif via seleksi PCA untuk mengklasifikasi 5 kelas bentuk wajah 
                dan merekomendasikan siluet kacamata penyeimbang proporsi secara objektif.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 5 Kelas Bentuk Wajah & Prinsip Kacamata ────────────────
    st.markdown('<div class="section-title">5 Kategori Bentuk Wajah &amp; Prinsip Pemilihan Kacamata</div>', unsafe_allow_html=True)

    shapes = [
        ("Round (Bulat)", "Panjang dan lebar wajah seimbang, sudut rahang membulat lembut.", "Bingkai Persegi / Bersudut (Square, Rectangular, Wayfarer)", "Memberikan garis tegas dan efek kontras yang merampingkan siluet pipi."),
        ("Square (Kotak)", "Garis rahang tegas berotot, lebar dahi dan rahang hampir sama.", "Bingkai Melengkung / Bulat (Round, Oval, Cat-Eye halus)", "Sudut lengkung melembutkan ketegasan garis rahang yang kaku."),
        ("Oval (Lonjong)", "Proporsi alami seimbang dengan dagu sedikit lebih sempit dari dahi.", "Hampir Semua Siluet Bingkai (Retro Square, Walnut, Aviator)", "Bentuk wajah paling fleksibel karena proporsi simetris alaminya."),
        ("Heart (Hati)", "Dahi dan tulang pipi lebar dengan dagu lancip mengecil.", "Bingkai Aviator, Oval Lembut, Bottom-Heavy (aksen di bawah)", "Menyeimbangkan dahi yang lebar tanpa menambah beban di bagian atas."),
        ("Oblong (Panjang)", "Panjang vertikal wajah dominan lebih besar dibanding lebarnya.", "Bingkai Oversized, Tall Frame (lensa tinggi), Bingkai Tebal", "Tinggi lensa yang besar memotong kesan wajah panjang agar lebih proporsional.")
    ]

    for title, char, rec, reason in shapes:
        st.markdown(f"""
        <div class="clean-card" style="margin-bottom: 0.6rem; padding: 1rem 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.3rem;">
                <div style="font-weight: 700; color: var(--text); font-size: 0.95rem;">
                    Bentuk {title}
                </div>
                <div style="font-size: 0.78rem; font-weight: 600; color: var(--accent);">
                    Rekomendasi: {rec}
                </div>
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.5;">
                <span style="color: var(--text);">Karakteristik:</span> {char} &bull; <span style="color: var(--text);">Alasan:</span> {reason}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Status Lingkungan Sistem ──────────────────────────────
    st.markdown('<div class="section-title">Status Lingkungan Komputasi</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Dataset Pelatihan</div>
            <div class="clean-metric-val">{DATASET_SAMPLES_COUNT} Data</div>
            <div class="clean-metric-desc">face_shape_features.csv</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Face Landmarker</div>
            <div class="clean-metric-val">478 Titik 3D</div>
            <div class="clean-metric-desc">MediaPipe Tasks Visi</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">OpenCV Engine</div>
            <div class="clean-metric-val">v{cv2.__version__}</div>
            <div class="clean-metric-desc">Image Processing Library</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Python Environment</div>
            <div class="clean-metric-val">{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}</div>
            <div class="clean-metric-desc">Runtime System</div>
        </div>
        """, unsafe_allow_html=True)
