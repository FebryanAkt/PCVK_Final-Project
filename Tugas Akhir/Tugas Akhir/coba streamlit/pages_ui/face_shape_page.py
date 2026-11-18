import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from utils.cv_helpers import load_image
from utils.face_shape_analyzer import (
    analyze_face_shape_and_glasses,
    create_geometric_visualization
)
from config import BASE_DIR

REF_81_IMG = BASE_DIR / "assets" / "sample_faces" / "dlib_81_reference.jpg"
SAMPLE_FACE_1 = BASE_DIR / "assets" / "sample_faces" / "sample_face_1.jpg"
SAMPLE_FACE_2 = BASE_DIR / "assets" / "sample_faces" / "sample_face_2.jpg"

def render_face_shape_page():
    # Header Jurnal & GitHub
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <span class="vl-badge vl-badge-info" style="margin-bottom: 0.5rem;">Dlib 81 Landmarks & Rekomendasi Ukuran Kacamata</span>
        <h2 style="font-weight: 800; color: #0F172A; margin-top: 0.2rem; margin-bottom: 0.2rem; letter-spacing: -0.02em;">
            Ekstraksi 81 Landmark Wajah, Pengukuran Geometri, dan Rekomendasi Kacamata
        </h2>
        <div style="color: #64748B; font-size: 0.92rem; line-height: 1.5;">
            Diintegrasikan dari model <strong>codeniko/shape_predictor_81_face_landmarks</strong> (13 titik dahi tambahan + 68 titik standar) 
            dan metodologi perhitungan jarak anatomis jurnal <em>"Human Face Shape Classification with Machine Learning"</em> (Mehta & Mahmoud)
            untuk menghasilkan rekomendasi ukuran spesifikasi kacamata (Frame Width, Lens Width, Bridge, dan Temple Length).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Input Citra
    st.sidebar.markdown("**Pilihan Citra Input**")
    source_mode = st.sidebar.radio(
        "Pilih Sumber Foto:",
        [
            "Citra Referensi 81 Titik (Scarlett Johansson)",
            "Contoh Citra 1 (R.A. Kartini)",
            "Contoh Citra 2 (Michael Jordan)",
            "Unggah Foto Wajah Sendiri"
        ],
        label_visibility="collapsed",
        index=0
    )

    min_conf = st.sidebar.slider(
        "Confidence Threshold Deteksi:",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    img_rgb = None

    if source_mode == "Citra Referensi 81 Titik (Scarlett Johansson)":
        if REF_81_IMG.exists():
            img_bgr = cv2.imread(str(REF_81_IMG))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.sidebar.caption("Memuat foto referensi 81 titik Dlib (codeniko).")
    elif source_mode == "Contoh Citra 1 (R.A. Kartini)":
        if SAMPLE_FACE_1.exists():
            img_bgr = cv2.imread(str(SAMPLE_FACE_1))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.sidebar.caption("Memuat contoh citra: R.A. Kartini.")
    elif source_mode == "Contoh Citra 2 (Michael Jordan)":
        if SAMPLE_FACE_2.exists():
            img_bgr = cv2.imread(str(SAMPLE_FACE_2))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.sidebar.caption("Memuat contoh citra: Michael Jordan.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Unggah foto tampak depan (Front-Profile):",
            type=["jpg", "jpeg", "png"],
            key="face_shape_uploader_v2"
        )
        if uploaded_file is not None:
            img_rgb = load_image(uploaded_file)
        else:
            st.info("Silakan unggah foto wajah lurus tampak depan melalui bilah sisi kiri, atau gunakan contoh citra bawaan.")
            return

    if img_rgb is None:
        st.error("Gagal memuat citra input.")
        return

    # Pemrosesan dan Ekstraksi
    with st.spinner("Mengekstrak 81 koordinat landmark wajah dan mengalkulasi ukuran geometri..."):
        annotated_81, res = analyze_face_shape_and_glasses(img_rgb, min_conf=min_conf)

    if annotated_81 is None:
        st.warning(f"Keterangan: {res}")
        st.image(img_rgb, caption="Citra Input Asli", use_container_width=True)
        return

    # Visualisasi Utama
    st.markdown('<div class="vl-section-header">Visualisasi 81 Titik Landmark Wajah (Codeniko / Dlib 81)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="vl-stat-label">Citra Masukan Asli</div>', unsafe_allow_html=True)
        st.image(img_rgb, caption=f"Resolusi: {img_rgb.shape[1]}x{img_rgb.shape[0]} px", use_container_width=True)
    with c2:
        st.markdown('<div class="vl-stat-label">Hasil Anotasi 81 Titik Landmark & Nomor Indeks (0 - 80)</div>', unsafe_allow_html=True)
        st.image(annotated_81, caption="Titik Kuning dan Angka Merah (Persis Model Dlib-81 Codeniko)", use_container_width=True)

    # 1. KARTU REKOMENDASI UKURAN KACAMATA (HASIL UTAMA)
    gl = res["glasses"]
    pred_shape = res["predicted_shape"]
    prob = res["probabilities"].get(pred_shape, 0.0)

    st.markdown(f"""
    <div class="vl-hero" style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 65%, #2563EB 100%); padding: 1.8rem 2.2rem; margin-top: 1rem; margin-bottom: 1.5rem;">
        <span class="vl-badge vl-badge-info" style="background: rgba(255, 255, 255, 0.15); color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 0.6rem;">
            Rekomendasi Spesifikasi Kacamata Presisi
        </span>
        <div style="font-size: 2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
            UKURAN FRAME: {gl['size_category']}
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 0.8rem; align-items: baseline;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #93C5FD;">
                Kode Standar Optik: <code>{gl['optical_code']}</code>
            </div>
            <div style="font-size: 0.95rem; color: #E2E8F0;">
                Bentuk Wajah Terdeteksi: <strong>{pred_shape.upper()} ({prob}%)</strong>
            </div>
        </div>
        <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.5rem;">
            Rekomendasi dihitung dari pengukuran proporsi Lebar Pelipis (D2), Lebar Tulang Pipi (D1), dan Lebar Jembatan Hidung (titik 39-42).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Rincian Dimensi Kacamata Standar Optik
    st.markdown('<div class="vl-section-header">Rincian Ukuran Bingkai Kacamata Rekomendasi</div>', unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">Lebar Bingkai Total</div>
            <div class="vl-stat-val" style="color: #2563EB;">{gl['recommended_frame_width']} mm</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Total Frame Width</div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">Lebar Lensa (Lens Width)</div>
            <div class="vl-stat-val" style="color: #2563EB;">{gl['lens_width']}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Horizontal Lens Size (A)</div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">Jembatan Hidung (Bridge)</div>
            <div class="vl-stat-val" style="color: #2563EB;">{gl['bridge_width'].split('(')[-1].replace(')', '')}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Distance Between Lenses</div>
        </div>
        """, unsafe_allow_html=True)
    with g4:
        st.markdown(f"""
        <div class="vl-stat-box">
            <div class="vl-stat-label">Panjang Gagang (Temple)</div>
            <div class="vl-stat-val" style="color: #2563EB;">{gl['temple_arm_length']}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Temple Arm Length</div>
        </div>
        """, unsafe_allow_html=True)

    # Model Frame yang Cocok Berdasarkan Bentuk Wajah
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
        <div class="vl-card" style="border-left: 4px solid #16A34A;">
            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.3rem;">Bentuk Bingkai Kacamata yang Direkomendasikan</div>
            <div style="font-size: 0.85rem; color: #166534; background: #DCFCE7; padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 0.5rem;">
                {gl['best_frames']}
            </div>
            <div style="font-size: 0.8rem; color: #475569;">
                <strong>Alasan Estetis:</strong> {gl['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="vl-card" style="border-left: 4px solid #DC2626;">
            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.3rem;">Bentuk Bingkai yang Perlu Dihindari</div>
            <div style="font-size: 0.85rem; color: #991B1B; background: #FEE2E2; padding: 0.6rem 0.9rem; border-radius: 6px; margin-bottom: 0.5rem;">
                {gl['avoid_frames']}
            </div>
            <div style="font-size: 0.8rem; color: #475569;">
                <strong>Pedoman Tinggi Lensa:</strong> {gl['lens_height']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. HASIL PERHITUNGAN JARAK ANATOMIS (SESUAI RUMUS JURNAL)
    st.markdown('<div class="vl-section-header">Hasil Perhitungan Jarak Anatomis Wajah (Formula Jurnal)</div>', unsafe_allow_html=True)
    
    px = res["measurements_px"]
    mm = res["measurements_mm"]
    r = res["ratios"]

    meas_data = [
        {
            "Nama Pengukuran Wajah": "Lebar Pelipis / Dahi (Temple Width)",
            "Titik Landmark Dlib": "Titik 75 ke 79 (D2)",
            "Formula / Deskripsi": "Jarak horizontal antara batas pelipis kanan dan kiri",
            "Nilai Piksel": f"{px['D2']:.1f} px",
            "Estimasi Milimeter": f"{mm['Temple_Width_mm']} mm"
        },
        {
            "Nama Pengukuran Wajah": "Lebar Pipi / Tulang Pipi (Cheekbone Width)",
            "Titik Landmark Dlib": "Titik 2 ke 14 (D1)",
            "Formula / Deskripsi": "Jarak terluar antara tulang pipi kiri dan kanan",
            "Nilai Piksel": f"{px['D1']:.1f} px",
            "Estimasi Milimeter": f"{mm['Cheekbone_Width_mm']} mm"
        },
        {
            "Nama Pengukuran Wajah": "Panjang Vertikal Wajah (Face Length)",
            "Titik Landmark Dlib": "Midpoint 69 & 72 ke Titik 8 (D3)",
            "Formula / Deskripsi": "Jarak dari garis rambut atas ke puncak dagu bawah",
            "Nilai Piksel": f"{px['D3']:.1f} px",
            "Estimasi Milimeter": f"{mm['Face_Length_mm']} mm"
        },
        {
            "Nama Pengukuran Wajah": "Lebar Sudut Rahang (Jaw Width)",
            "Titik Landmark Dlib": "Titik 4 ke 12 (D5)",
            "Formula / Deskripsi": "Jarak bilateral antara kedua sudut gonion rahang",
            "Nilai Piksel": f"{px['D5']:.1f} px",
            "Estimasi Milimeter": f"{mm['Jaw_Width_mm']} mm"
        },
        {
            "Nama Pengukuran Wajah": "Lebar Rahang Bawah (Lower Jaw Width)",
            "Titik Landmark Dlib": "Titik 6 ke 10 (D6)",
            "Formula / Deskripsi": "Lebar rahang bawah di area sekitar mulut",
            "Nilai Piksel": f"{px['D6']:.1f} px",
            "Estimasi Milimeter": f"{round(px['D6'] * (mm['Cheekbone_Width_mm']/px['D1']), 1)} mm"
        },
        {
            "Nama Pengukuran Wajah": "Lebar Dasar Dagu (Chin Base Width)",
            "Titik Landmark Dlib": "Titik 7 ke 9 (D7)",
            "Formula / Deskripsi": "Lebar dasar dagu (ketajaman vs kekotakan dagu)",
            "Nilai Piksel": f"{px['D7']:.1f} px",
            "Estimasi Milimeter": f"{round(px['D7'] * (mm['Cheekbone_Width_mm']/px['D1']), 1)} mm"
        },
        {
            "Nama Pengukuran Wajah": "Lebar Jembatan Hidung (Nose Bridge)",
            "Titik Landmark Dlib": "Titik 39 ke 42",
            "Formula / Deskripsi": "Jarak sudut dalam mata kanan ke sudut dalam mata kiri",
            "Nilai Piksel": f"{px['Bridge_Width']:.1f} px",
            "Estimasi Milimeter": f"{mm['Bridge_Width_mm']} mm"
        },
        {
            "Nama Pengukuran Wajah": "Jarak Antarpupil (IPD)",
            "Titik Landmark Dlib": "Center mata 36-41 ke 42-47",
            "Formula / Deskripsi": "Jarak pupil mata (acuan kalibrasi antropometri optik)",
            "Nilai Piksel": f"{px['IPD']:.1f} px",
            "Estimasi Milimeter": "63.0 mm (Acuan Standar)"
        }
    ]

    st.dataframe(meas_data, use_container_width=True, hide_index=True)

    # 3. RASIO GEOMETRIS & SUDUT
    with st.expander("Lihat 10 Rasio Geometri (R1 - R10) & 3 Sudut (A1 - A3) Sesuai Jurnal"):
        col_r, col_a = st.columns([2, 1])
        with col_r:
            st.markdown("**10 Rasio Proporsi Wajah (R1 - R10):**")
            st.write({
                "R1 (D2/D1 - Rasio Dahi/Pipi)": f"{r['R1']:.4f}",
                "R2 (D1/D3 - Rasio Lebar/Panjang)": f"{r['R2']:.4f}",
                "R3 (D2/D3 - Rasio Dahi/Panjang)": f"{r['R3']:.4f}",
                "R4 (D1/D5 - Rasio Pipi/Rahang)": f"{r['R4']:.4f}",
                "R5 (D6/D5 - Rasio Bawah/Rahang)": f"{r['R5']:.4f}",
                "R6 (D4/D6 - Rasio Garis Rahang/Bawah)": f"{r['R6']:.4f}",
                "R7 (D6/D1 - Rasio Bawah/Pipi)": f"{r['R7']:.4f}",
                "R8 (D5/D2 - Rasio Rahang/Dahi)": f"{r['R8']:.4f}",
                "R9 (D4/D5 - Rasio Garis/Lebar Rahang)": f"{r['R9']:.4f}",
                "R10 (D7/D6 - Rasio Dagu/Bawah)": f"{r['R10']:.4f}"
            })
        with col_a:
            st.markdown("**3 Sudut Ketegasan Rahang:**")
            for k, v in res["angles"].items():
                st.write(f"- {k}: {v:.2f} derajat")

    # 4. TABEL LENGKAP SELURUH 81 KOORDINAT LANDMARK (X, Y)
    with st.expander("Tabel Seluruh 81 Titik Koordinat Landmark Wajah (X, Y)"):
        st.caption("Menampilkan posisi koordinat piksel dari titik 0 hingga 80 berdasarkan codeniko/shape_predictor_81_face_landmarks:")
        st.dataframe(res["points_table"], use_container_width=True, hide_index=True)
