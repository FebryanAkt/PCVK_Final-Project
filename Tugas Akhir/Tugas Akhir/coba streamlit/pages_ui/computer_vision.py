import streamlit as st
import numpy as np
import cv2
from utils.cv_helpers import load_image
from utils.model_loader import VisionModelManager
from config import FACE_LANDMARKER_PATH, HAND_LANDMARKER_PATH

def render_cv_page():
    # Header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-weight: 800; color: #0F172A; margin-bottom: 0.2rem; letter-spacing: -0.02em;">
            Modul Computer Vision dan Deep Learning
        </h2>
        <div style="color: #64748B; font-size: 0.92rem;">
            Inferensi model machine learning untuk ekstraksi titik koordinat landmark anatomis (wajah dan tangan) serta deteksi objek.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Upload & Parameters
    st.sidebar.markdown("**Input Citra Pengujian**")
    uploaded_file = st.sidebar.file_uploader(
        "Unggah foto wajah atau tangan (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        key="cv_uploader"
    )

    min_conf = st.sidebar.slider("Tingkat Keyakinan Minimum (Confidence):", 0.1, 1.0, 0.5, step=0.05)

    if uploaded_file is None:
        st.info("Silakan unggah citra yang memuat objek wajah atau tangan melalui panel bilah sisi kiri untuk memulai inferensi.")
        
        # Display model availability status cards
        st.markdown('<div class="vl-section-header">Status Kesiapan Model Eksternal</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if VisionModelManager.is_face_model_available():
                st.markdown(f"""
                <div class="vl-card" style="border-left: 4px solid #16A34A;">
                    <div style="font-weight: 700; color: #0F172A;">MediaPipe Face Landmarker</div>
                    <div style="font-size: 0.8rem; color: #15803D; margin-top: 0.2rem;">Model terpasang: {FACE_LANDMARKER_PATH.name}</div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem;">Siap mendeteksi 478 koordinat titik 3D pada kontur wajah, mata, alis, dan bibir.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="vl-card" style="border-left: 4px solid #DC2626;">
                    <div style="font-weight: 700; color: #0F172A;">MediaPipe Face Landmarker</div>
                    <div style="font-size: 0.8rem; color: #B91C1C; margin-top: 0.2rem;">File model belum ditemukan.</div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            if VisionModelManager.is_hand_model_available():
                st.markdown(f"""
                <div class="vl-card" style="border-left: 4px solid #16A34A;">
                    <div style="font-weight: 700; color: #0F172A;">MediaPipe Hand Landmarker</div>
                    <div style="font-size: 0.8rem; color: #15803D; margin-top: 0.2rem;">Model terpasang: {HAND_LANDMARKER_PATH.name}</div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem;">Siap mendeteksi 21 titik sendi jari tangan dan klasifikasi orientasi tangan.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="vl-card" style="border-left: 4px solid #DC2626;">
                    <div style="font-weight: 700; color: #0F172A;">MediaPipe Hand Landmarker</div>
                    <div style="font-size: 0.8rem; color: #B91C1C; margin-top: 0.2rem;">File model belum ditemukan.</div>
                </div>
                """, unsafe_allow_html=True)
        return

    # Process loaded image
    img_rgb = load_image(uploaded_file)

    # Tabs for CV models
    tab_face, tab_hand, tab_haar = st.tabs([
        "Face Landmarker (478 Titik 3D)",
        "Hand Landmarker (21 Titik Sendi)",
        "OpenCV Haar Cascade"
    ])

    # TAB 1: Face Landmarker
    with tab_face:
        max_faces = st.slider("Maksimal Jumlah Wajah yang Dideteksi:", 1, 6, 2, key="face_max")
        if not VisionModelManager.is_face_model_available():
            st.error("Model face_landmarker.task tidak ditemukan pada sistem.")
        else:
            with st.spinner("Memproses inferensi landmark wajah..."):
                try:
                    annotated_img, info = VisionModelManager.detect_face_landmarks(
                        img_rgb, num_faces=max_faces, min_conf=min_conf
                    )
                    render_cv_result(
                        source_img=img_rgb,
                        output_img=annotated_img,
                        title="Ekstraksi 478 Landmark Wajah",
                        stats=[
                            ("Wajah Terdeteksi", f"{info.get('faces_detected', 0)} Objek"),
                            ("Confidence Threshold", f"{min_conf:.2f}"),
                            ("Status Deteksi", "Sukses" if info.get('faces_detected', 0) > 0 else "Nol Wajah")
                        ]
                    )
                except Exception as e:
                    st.error(f"Gagal melakukan inferensi: {e}")

    # TAB 2: Hand Landmarker
    with tab_hand:
        max_hands = st.slider("Maksimal Jumlah Tangan yang Dideteksi:", 1, 4, 2, key="hand_max")
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
                        title="Ekstraksi 21 Landmark Artikulasi Tangan",
                        stats=[
                            ("Tangan Terdeteksi", f"{info.get('hands_detected', 0)} Tangan"),
                            ("Klasifikasi Tangan", handedness_str),
                            ("Confidence Threshold", f"{min_conf:.2f}")
                        ]
                    )
                except Exception as e:
                    st.error(f"Gagal melakukan inferensi: {e}")

    # TAB 3: Haar Cascade
    with tab_haar:
        st.markdown('<div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.8rem;">Deteksi wajah menggunakan metode klasik Viola-Jones (Haar Feature-based Cascade Classifier).</div>', unsafe_allow_html=True)
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
    # Metric Pills
    stat_cols = st.columns(len(stats))
    for col, (label, val) in zip(stat_cols, stats):
        with col:
            st.markdown(f"""
            <div class="vl-stat-box" style="padding: 0.8rem 1rem;">
                <div class="vl-stat-label">{label}</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(source_img, caption="Citra Input Asli", use_container_width=True)
    with c2:
        st.image(output_img, caption=f"Anotasi Model: {title}", use_container_width=True)
