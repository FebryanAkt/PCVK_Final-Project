import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from utils.cv_helpers import load_image
from utils.face_shape_analyzer import analyze_face_shape_and_glasses
from config import BASE_DIR

REF_81_IMG    = BASE_DIR / "assets" / "sample_faces" / "dlib_81_reference.jpg"
SAMPLE_FACE_1 = BASE_DIR / "assets" / "sample_faces" / "sample_face_1.jpg"
SAMPLE_FACE_2 = BASE_DIR / "assets" / "sample_faces" / "sample_face_2.jpg"

def render_face_shape_page():
    # ── Header Ringkas ─────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
            Klasifikasi Bentuk Wajah &amp; Rekomendasi Kacamata
        </h2>
        <div style="color: var(--text-muted); font-size: 0.88rem; line-height: 1.5;">
            Ekstraksi landmark geometris Dlib untuk mengklasifikasikan 5 bentuk wajah 
            (Heart, Oblong, Oval, Round, Square) dan menentukan siluet kacamata yang proporsional 
            sesuai metodologi jurnal Mehta &amp; Mahmoud.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Bar Bersih ──────────────────────────────────────
    col_input, col_conf = st.columns([3, 1.2])
    with col_input:
        source_mode = st.radio(
            "Pilih Citra Masukan:",
            [
                "Foto Referensi 81 Titik",
                "Contoh: R.A. Kartini",
                "Contoh: Michael Jordan",
                "Unggah Foto Sendiri"
            ],
            horizontal=True,
            index=0,
            label_visibility="collapsed"
        )
    with col_conf:
        min_conf = st.slider("Confidence Deteksi:", 0.1, 1.0, 0.5, step=0.05)

    img_rgb = None

    if "Referensi" in source_mode:
        if REF_81_IMG.exists():
            img_bgr = cv2.imread(str(REF_81_IMG))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif "Kartini" in source_mode:
        if SAMPLE_FACE_1.exists():
            img_bgr = cv2.imread(str(SAMPLE_FACE_1))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif "Jordan" in source_mode:
        if SAMPLE_FACE_2.exists():
            img_bgr = cv2.imread(str(SAMPLE_FACE_2))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        uploaded_file = st.file_uploader(
            "Pilih foto wajah lurus tampak depan (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
            key="face_uploader_simple"
        )
        if uploaded_file is not None:
            img_rgb = load_image(uploaded_file)
        else:
            st.info("Silakan unggah foto wajah lurus tampak depan, atau gunakan salah satu foto contoh di atas.")
            return

    if img_rgb is None:
        st.error("Gagal memuat citra input.")
        return

    # ── Ekstraksi Landmark & Prediksi ─────────────────────────
    with st.spinner("Mengekstrak 81 landmark dan menghitung geometri wajah..."):
        annotated_81, res = analyze_face_shape_and_glasses(img_rgb, min_conf=min_conf)

    if annotated_81 is None:
        st.warning(f"Keterangan sistem: {res}")
        st.image(img_rgb, caption="Citra Asli", use_container_width=True)
        return

    # ── 1. Visualisasi Citra & Anotasi Landmark ───────────────
    st.markdown('<div class="section-title">Visualisasi Citra &amp; 81 Landmark Wajah</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        st.image(img_rgb, caption=f"Citra Masukan ({img_rgb.shape[1]} × {img_rgb.shape[0]} px)", use_container_width=True)
    with v2:
        st.image(annotated_81, caption="Anotasi 81 Titik Landmark Dlib (Indeks 0 - 80)", use_container_width=True)

    # ── 2. Hasil Klasifikasi Bentuk Wajah ─────────────────────
    gl = res["glasses"]
    pred_shape = res["predicted_shape"]
    prob = res["probabilities"].get(pred_shape, 0.0)

    st.markdown('<div class="section-title">Hasil Prediksi Bentuk Wajah</div>', unsafe_allow_html=True)

    # Kartu Prediksi Utama
    st.markdown(f"""
    <div class="clean-card" style="border-left: 4px solid var(--accent); margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">
                    Bentuk Wajah Terklasifikasi
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em;">
                    {pred_shape.upper()}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">
                    Tingkat Keyakinan (Confidence)
                </div>
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--accent);">
                    {prob}%
                </div>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.4rem;">
            Model: <em>{res.get('model_source', 'Machine Learning Ensemble Soft Voting')}</em> &bull; Terlatih pada 596 sampel citra wajah tugas akhir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Distribusi Probabilitas 5 Kelas
    p_cols = st.columns(5)
    for idx, (cls_name, conf_val) in enumerate(res["probabilities"].items()):
        is_pred = (cls_name.lower() == pred_shape.lower())
        with p_cols[idx]:
            act_cls = "active" if is_pred else ""
            st.markdown(f"""
            <div class="conf-box {act_cls}">
                <div class="conf-name">{cls_name}</div>
                <div class="conf-val">{conf_val}%</div>
                <div style="background: var(--border); height: 4px; border-radius: 9999px; overflow: hidden; margin-top: 0.35rem;">
                    <div style="background: {'var(--accent)' if is_pred else 'var(--text-light)'}; height: 100%; width: {min(100, conf_val)}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 3. Rekomendasi Kacamata Sesuai Proporsi Wajah ─────────
    st.markdown('<div class="section-title">Rekomendasi Bentuk &amp; Ukuran Kacamata</div>', unsafe_allow_html=True)
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="clean-card" style="border-left: 3px solid var(--success); height: 100%;">
            <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--success); margin-bottom: 0.4rem;">
                ✓ Siluet Bingkai yang Cocok
            </div>
            <div style="font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem;">
                {gl['best_frames']}
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.55;">
                <strong>Alasan Proporsi:</strong> {gl['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="clean-card" style="border-left: 3px solid #DC2626; height: 100%;">
            <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #DC2626; margin-bottom: 0.4rem;">
                ✕ Bingkai yang Perlu Dihindari
            </div>
            <div style="font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem;">
                {gl['avoid_frames']}
            </div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.55;">
                <strong>Panduan Tinggi Lensa:</strong> {gl['lens_height']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 Spesifikasi Dimensi Kacamata
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Lebar Bingkai Total</div>
            <div class="clean-metric-val">{gl['recommended_frame_width']} mm</div>
            <div class="clean-metric-desc">Kategori: {gl['size_category']}</div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Lebar Lensa (A)</div>
            <div class="clean-metric-val">{gl['lens_width']}</div>
            <div class="clean-metric-desc">Horizontal Lens Width</div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Jembatan Hidung</div>
            <div class="clean-metric-val">{gl['bridge_width'].split('(')[-1].replace(')', '')}</div>
            <div class="clean-metric-desc">Distance Between Lenses</div>
        </div>
        """, unsafe_allow_html=True)
    with g4:
        st.markdown(f"""
        <div class="clean-metric">
            <div class="clean-metric-label">Kode Standar Optik</div>
            <div class="clean-metric-val" style="font-size: 1.15rem; font-family: var(--font-mono);">{gl['optical_code']}</div>
            <div class="clean-metric-desc">Panjang Gagang: {gl['temple_arm_length']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Pengukuran Geometris Wajah (Mehta & Mahmoud) ───────
    st.markdown('<div class="section-title">Hasil Pengukuran Jarak Geometris (Formula Jurnal)</div>', unsafe_allow_html=True)
    
    px = res["measurements_px"]
    mm = res["measurements_mm"]
    r = res["ratios"]

    meas_data = [
        {
            "Parameter Antropometri": "Lebar Tulang Pipi (Cheekbone Width / D1)",
            "Titik Kunci Dlib": "Titik 2 ke 14",
            "Nilai Piksel": f"{px['D1']:.1f} px",
            "Estimasi Milimeter": f"{mm['Cheekbone_Width_mm']} mm"
        },
        {
            "Parameter Antropometri": "Lebar Pelipis / Dahi (Temple Width / D2)",
            "Titik Kunci Dlib": "Titik 75 ke 79",
            "Nilai Piksel": f"{px['D2']:.1f} px",
            "Estimasi Milimeter": f"{mm['Temple_Width_mm']} mm"
        },
        {
            "Parameter Antropometri": "Panjang Vertikal Wajah (Face Length / D3)",
            "Titik Kunci Dlib": "Midpoint 69/72 ke Titik 8",
            "Nilai Piksel": f"{px['D3']:.1f} px",
            "Estimasi Milimeter": f"{mm['Face_Length_mm']} mm"
        },
        {
            "Parameter Antropometri": "Lebar Sudut Rahang (Jaw Width / D5)",
            "Titik Kunci Dlib": "Titik 4 ke 12 (Gonion)",
            "Nilai Piksel": f"{px['D5']:.1f} px",
            "Estimasi Milimeter": f"{mm['Jaw_Width_mm']} mm"
        },
        {
            "Parameter Antropometri": "Lebar Rahang Bawah (Lower Jaw Width / D6)",
            "Titik Kunci Dlib": "Titik 6 ke 10",
            "Nilai Piksel": f"{px['D6']:.1f} px",
            "Estimasi Milimeter": f"{round(px['D6'] * (mm['Cheekbone_Width_mm']/px['D1']), 1)} mm"
        },
        {
            "Parameter Antropometri": "Lebar Dasar Dagu (Chin Base Width / D7)",
            "Titik Kunci Dlib": "Titik 7 ke 9",
            "Nilai Piksel": f"{px['D7']:.1f} px",
            "Estimasi Milimeter": f"{round(px['D7'] * (mm['Cheekbone_Width_mm']/px['D1']), 1)} mm"
        },
        {
            "Parameter Antropometri": "Lebar Jembatan Hidung (Nose Bridge)",
            "Titik Kunci Dlib": "Titik 39 ke 42",
            "Nilai Piksel": f"{px['Bridge_Width']:.1f} px",
            "Estimasi Milimeter": f"{mm['Bridge_Width_mm']} mm"
        }
    ]
    st.dataframe(meas_data, use_container_width=True, hide_index=True)

    # ── 5. Detail Akademik: 10 Rasio & 81 Titik Koordinat ─────
    with st.expander("Detail Akademik: 10 Rasio Geometri (R1 - R10) & 3 Sudut Rahang"):
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
                st.write(f"- {k}: **{v:.2f}°**")

    with st.expander("Tabel Lengkap 81 Koordinat Landmark Wajah (X, Y)"):
        st.caption("Koordinat piksel hasil pemetaan 81 titik Dlib (Codeniko model):")
        st.dataframe(res["points_table"], use_container_width=True, hide_index=True)
