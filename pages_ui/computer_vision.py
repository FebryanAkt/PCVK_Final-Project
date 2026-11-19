import streamlit as st
import numpy as np
import cv2
from utils.cv_helpers import load_image
from utils.model_loader import VisionModelManager
from utils.face_shape_analyzer import analyze_face_shape_and_glasses
from config import FACE_LANDMARKER_PATH, HAND_LANDMARKER_PATH

def render_cv_page():
    # ── Header Ringkas ─────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
            Modul Computer Vision &amp; Deep Learning
        </h2>
        <div style="color: var(--text-muted); font-size: 0.88rem; line-height: 1.5;">
            Inferensi model machine learning untuk ekstraksi 478 titik landmark 3D wajah, 
            21 sendi jari tangan (MediaPipe), serta deteksi wajah klasik OpenCV Haar Cascade.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload & Slider ───────────────────────────────────────
    c_up, c_conf = st.columns([3, 1.2])
    with c_up:
        uploaded_file = st.file_uploader(
            "Unggah foto wajah atau tangan (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
            key="cv_uploader_clean"
        )
    with c_conf:
        min_conf = st.slider("Confidence Min:", 0.1, 1.0, 0.5, step=0.05)

    if uploaded_file is None:
        st.info("Silakan unggah foto yang memuat wajah atau tangan melalui panel di atas untuk memulai deteksi landmark.")
        
        # Status Model
        st.markdown('<div class="section-title">Status Kesiapan Model Eksternal</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if VisionModelManager.is_face_model_available():
                st.markdown(f"""
                <div class="clean-card" style="border-left: 3px solid var(--success);">
                    <div style="font-weight: 700; color: var(--text); font-size: 0.95rem;">MediaPipe Face Landmarker 3D</div>
                    <div style="font-size: 0.8rem; color: var(--success); margin-top: 0.2rem;">✓ Model aktif: {FACE_LANDMARKER_PATH.name}</div>
                    <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem;">
                        Mendeteksi 478 koordinat 3D pada kontur wajah, mata, alis, dan bibir.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="clean-card" style="border-left: 3px solid #DC2626;">
                    <div style="font-weight: 700; color: #DC2626;">Model Face Landmarker Belum Ditemukan</div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            if VisionModelManager.is_hand_model_available():
                st.markdown(f"""
                <div class="clean-card" style="border-left: 3px solid var(--success);">
                    <div style="font-weight: 700; color: var(--text); font-size: 0.95rem;">MediaPipe Hand Landmarker 3D</div>
                    <div style="font-size: 0.8rem; color: var(--success); margin-top: 0.2rem;">✓ Model aktif: {HAND_LANDMARKER_PATH.name}</div>
                    <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem;">
                        Mendeteksi 21 titik sendi jari tangan dan orientasi kiri / kanan.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="clean-card" style="border-left: 3px solid #DC2626;">
                    <div style="font-weight: 700; color: #DC2626;">Model Hand Landmarker Belum Ditemukan</div>
                </div>
                """, unsafe_allow_html=True)
        return

    # Process loaded image
    img_rgb = load_image(uploaded_file)

    tab_face, tab_hand, tab_haar = st.tabs([
        "Face Landmarker (478 Titik 3D)",
        "Hand Landmarker (21 Sendi Tangan)",
        "OpenCV Haar Cascade"
    ])

    # TAB 1: Face Landmarker
    with tab_face:
        max_faces = st.slider("Maksimal Wajah yang Dideteksi:", 1, 6, 2, key="face_max_clean")
        if not VisionModelManager.is_face_model_available():
            st.error("Model face_landmarker.task tidak ditemukan pada sistem.")
        else:
            with st.spinner("Memproses inferensi landmark wajah 3D..."):
                try:
                    annotated_img, info = VisionModelManager.detect_face_landmarks(
                        img_rgb, num_faces=max_faces, min_conf=min_conf
                    )
                    render_cv_result(
                        source_img=img_rgb,
                        output_img=annotated_img,
                        title="Ekstraksi 478 Landmark Wajah 3D",
                        stats=[
                            ("Wajah Terdeteksi", f"{info.get('faces_detected', 0)} Objek"),
                            ("Confidence Min", f"{min_conf:.2f}"),
                            ("Status Deteksi", "Sukses" if info.get('faces_detected', 0) > 0 else "Nol Wajah")
                        ]
                    )

                    # Hasil Analisis Geometri Wajah persis seperti di section Uji Coba Kamera Langsung
                    if info.get('faces_detected', 0) > 0:
                        with st.spinner("Mengekstrak geometri wajah dan rekomendasi kacamata..."):
                            annotated_81, res = analyze_face_shape_and_glasses(img_rgb, min_conf=min_conf)
                            if res and isinstance(res, dict) and "predicted_shape" in res:
                                render_face_geometry_analysis(res)
                except Exception as e:
                    st.error(f"Gagal melakukan inferensi: {e}")

    # TAB 2: Hand Landmarker
    with tab_hand:
        max_hands = st.slider("Maksimal Tangan yang Dideteksi:", 1, 4, 2, key="hand_max_clean")
        if not VisionModelManager.is_hand_model_available():
            st.error("Model hand_landmarker.task tidak ditemukan pada sistem.")
        else:
            with st.spinner("Memproses inferensi landmark tangan..."):
                try:
                    annotated_img, info = VisionModelManager.detect_hand_landmarks(
                        img_rgb, num_hands=max_hands, min_conf=min_conf
                    )
                    handedness_str = ", ".join(info.get('handedness', [])) if info.get('handedness') else "Tidak ada"
                    render_cv_result(
                        source_img=img_rgb,
                        output_img=annotated_img,
                        title="Ekstraksi 21 Landmark Sendi Tangan",
                        stats=[
                            ("Tangan Terdeteksi", f"{info.get('hands_detected', 0)} Tangan"),
                            ("Klasifikasi Tangan", handedness_str),
                            ("Confidence Min", f"{min_conf:.2f}")
                        ]
                    )
                except Exception as e:
                    st.error(f"Gagal melakukan inferensi: {e}")

    # TAB 3: Haar Cascade
    with tab_haar:
        st.markdown('<div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.8rem;">Deteksi wajah menggunakan metode klasik Viola-Jones (Haar Feature-based Cascade Classifier).</div>', unsafe_allow_html=True)
        with st.spinner("Menjalankan deteksi Haar Cascade..."):
            try:
                annotated_img, info = VisionModelManager.detect_face_opencv(img_rgb)
                render_cv_result(
                    source_img=img_rgb,
                    output_img=annotated_img,
                    title="Deteksi Wajah Haar Cascade",
                    stats=[
                        ("Wajah Terdeteksi", f"{info.get('faces_detected', 0)} Objek"),
                        ("Metode", "OpenCV CascadeClassifier"),
                        ("Tipe Output", "Bounding Box (X, Y, W, H)")
                    ]
                )

                if info.get('faces_detected', 0) > 0:
                    with st.spinner("Mengekstrak geometri wajah dan rekomendasi kacamata..."):
                        annotated_81, res = analyze_face_shape_and_glasses(img_rgb, min_conf=min_conf)
                        if res and isinstance(res, dict) and "predicted_shape" in res:
                            render_face_geometry_analysis(res)
            except Exception as e:
                st.error(f"Gagal melakukan deteksi Haar Cascade: {e}")


def render_cv_result(source_img, output_img, title, stats):
    """Menampilkan hasil inferensi model CV dengan metrik status dan perbandingan 2 kolom."""
    stat_cols = st.columns(len(stats))
    for col, (label, val) in zip(stat_cols, stats):
        with col:
            st.markdown(f"""
            <div class="clean-metric">
                <div class="clean-metric-label">{label}</div>
                <div class="clean-metric-val" style="font-size: 1.25rem;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(source_img, caption="Citra Input Asli", use_container_width=True)
    with c2:
        st.image(output_img, caption=f"Hasil: {title}", use_container_width=True)


def render_face_geometry_analysis(res):
    """Menampilkan hasil analisis geometri wajah dan rekomendasi kacamata persis seperti pada webcam page."""
    gl = res["glasses"]
    pred_shape = res["predicted_shape"]
    prob = res["probabilities"].get(pred_shape, 0.0)

    st.markdown('<div class="section-title">Hasil Analisis Geometri Wajah Anda</div>', unsafe_allow_html=True)

    # Kartu Hasil Utama
    st.markdown(f"""
    <div class="clean-card" style="border-left: 4px solid var(--accent); margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">
                    Bentuk Wajah Terdeteksi
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: var(--text);">
                    {pred_shape.upper()}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">
                    Confidence Level
                </div>
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--accent);">
                    {prob}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Rekomendasi Kacamata
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="clean-card" style="border-left: 3px solid var(--success); height: 100%;">
            <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: var(--success); margin-bottom: 0.35rem;">
                ✓ Bentuk Bingkai yang Cocok
            </div>
            <div style="font-weight: 700; color: var(--text); font-size: 0.95rem; margin-bottom: 0.35rem;">
                {gl['best_frames']}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5;">
                <strong>Alasan:</strong> {gl['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="clean-card" style="border-left: 3px solid #DC2626; height: 100%;">
            <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #DC2626; margin-bottom: 0.35rem;">
                ✕ Bentuk Bingkai yang Perlu Dihindari
            </div>
            <div style="font-weight: 700; color: var(--text); font-size: 0.95rem; margin-bottom: 0.35rem;">
                {gl['avoid_frames']}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5;">
                <strong>Panduan Tinggi Lensa:</strong> {gl['lens_height']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 Metrik Ukuran
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

    # Tabel Jarak Geometri
    st.markdown('<div class="section-title">Hasil Pengukuran Jarak Geometris (Formula Jurnal)</div>', unsafe_allow_html=True)
    px = res["measurements_px"]
    mm = res["measurements_mm"]
    meas_data = [
        {"Parameter": "Lebar Tulang Pipi (D1)", "Titik Dlib": "Titik 2 ke 14", "Piksel": f"{px['D1']:.1f} px", "Estimasi mm": f"{mm['Cheekbone_Width_mm']} mm"},
        {"Parameter": "Lebar Pelipis / Dahi (D2)", "Titik Dlib": "Titik 75 ke 79", "Piksel": f"{px['D2']:.1f} px", "Estimasi mm": f"{mm['Temple_Width_mm']} mm"},
        {"Parameter": "Panjang Vertikal Wajah (D3)", "Titik Dlib": "Midpoint 69/72 ke 8", "Piksel": f"{px['D3']:.1f} px", "Estimasi mm": f"{mm['Face_Length_mm']} mm"},
        {"Parameter": "Lebar Sudut Rahang (D5)", "Titik Dlib": "Titik 4 ke 12 (Gonion)", "Piksel": f"{px['D5']:.1f} px", "Estimasi mm": f"{mm['Jaw_Width_mm']} mm"},
        {"Parameter": "Lebar Jembatan Hidung", "Titik Dlib": "Titik 39 ke 42", "Piksel": f"{px['Bridge_Width']:.1f} px", "Estimasi mm": f"{mm['Bridge_Width_mm']} mm"}
    ]
    st.dataframe(meas_data, use_container_width=True, hide_index=True)

    with st.expander("Tabel Lengkap 81 Titik Koordinat dari Tangkapan Kamera"):
        st.dataframe(res["points_table"], use_container_width=True, hide_index=True)
