import streamlit as st
import numpy as np
import cv2
from utils.cv_helpers import load_image
from utils.model_loader import VisionModelManager
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
