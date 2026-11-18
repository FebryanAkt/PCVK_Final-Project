import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image
from utils.cv_helpers import (
    load_image,
    to_grayscale,
    apply_blur,
    apply_edge_detection,
    apply_threshold,
    convert_color_space,
    plot_histogram
)

def create_sample_image():
    """Membuat citra sintetis uji coba dengan gradien dan geometri terdefinisi."""
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    for y in range(360):
        img[y, :, 0] = int(y / 360 * 240)
        img[y, :, 1] = int((360 - y) / 360 * 210)
        img[y, :, 2] = 180
    cv2.circle(img, (140, 180), 75, (255, 255, 255), -1)
    cv2.circle(img, (140, 180), 50, (37, 99, 235), -1)
    cv2.rectangle(img, (260, 100), (410, 260), (30, 41, 59), -1)
    cv2.line(img, (50, 310), (430, 310), (255, 255, 255), 3)
    cv2.putText(img, "VISION LAB", (110, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img

def render_image_processing_page():
    # ── Header Ringkas ─────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
            Modul Pengolahan Citra Digital (PCD)
        </h2>
        <div style="color: var(--text-muted); font-size: 0.88rem; line-height: 1.5;">
            Eksplorasi teknik transformasi spasial, penapisan frekuensi (blur), binarisasi adaptif, 
            deteksi tepi objek, dan analisis ruang warna citra secara interaktif.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Konfigurasi Sumber Citra ──────────────────────────────
    c_src, c_up = st.columns([1.5, 2.5])
    with c_src:
        source_choice = st.radio(
            "Pilih Sumber Citra:",
            ["Citra Geometri Bawaan", "Unggah Citra Sendiri"],
            index=0
        )

    img_rgb = None
    if "Unggah" in source_choice:
        with c_up:
            uploaded_file = st.file_uploader(
                "Format yang didukung: JPG, JPEG, PNG",
                type=["jpg", "jpeg", "png"]
            )
        if uploaded_file is not None:
            img_rgb = load_image(uploaded_file)
        else:
            img_rgb = create_sample_image()
    else:
        img_rgb = create_sample_image()

    # ── Metadata Citra Ringkas ────────────────────────────────
    h, w, c = img_rgb.shape if len(img_rgb.shape) == 3 else (*img_rgb.shape, 1)
    st.markdown(f"""
    <div style="background: var(--bg-subtle); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.5rem 1rem; margin: 0.8rem 0 1.2rem; display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; font-size: 0.8rem; color: var(--text-muted);">
        <div><strong style="color: var(--text);">Resolusi:</strong> {w} &times; {h} px</div>
        <div><strong style="color: var(--text);">Kanal:</strong> {c} ({'RGB' if c == 3 else 'Grayscale'})</div>
        <div><strong style="color: var(--text);">Rentang:</strong> [{int(np.min(img_rgb))}, {int(np.max(img_rgb))}]</div>
        <div><strong style="color: var(--text);">Tipe Data:</strong> {img_rgb.dtype}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs Operasi PCD ──────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Binarisasi & Threshold",
        "Penapisan (Blur)",
        "Deteksi Tepi (Edges)",
        "Ruang Warna",
        "Histogram"
    ])

    processed_img = None
    info_text = ""

    # TAB 1: Binarisasi
    with tab1:
        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            method = st.selectbox(
                "Metode Binarisasi:",
                ["Hanya Grayscale", "Binary Threshold", "Binary Inverted", "Otsu Automatic", "Adaptive Mean", "Adaptive Gaussian"]
            )
            
            if method in ["Binary Threshold", "Binary Inverted"]:
                thresh_val = st.slider("Nilai Ambang (Threshold):", 0, 255, 128)
                op_name = "Binary" if method == "Binary Threshold" else "Binary Inverted"
                processed_img = apply_threshold(img_rgb, op_name, thresh_val)
                info_text = f"{method} (T = {thresh_val})"
            elif method == "Hanya Grayscale":
                processed_img = to_grayscale(img_rgb)
                info_text = "Konversi Monokrom (Grayscale)"
            elif method == "Otsu Automatic":
                processed_img = apply_threshold(img_rgb, "Otsu")
                info_text = "Otsu Automatic Thresholding"
            elif method == "Adaptive Mean":
                processed_img = apply_threshold(img_rgb, "Adaptive Mean")
                info_text = "Adaptive Mean Thresholding"
            elif method == "Adaptive Gaussian":
                processed_img = apply_threshold(img_rgb, "Adaptive Gaussian")
                info_text = "Adaptive Gaussian Thresholding"

        with col_view:
            render_comparison(img_rgb, processed_img, info_text)

    # TAB 2: Smoothing & Blurring
    with tab2:
        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            filter_type = st.selectbox(
                "Pilihan Filter Spasial:",
                ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Averaging Box Blur"]
            )
            kernel = st.slider("Ukuran Kernel (Ganjil):", min_value=3, max_value=35, value=7, step=2)
            processed_img = apply_blur(img_rgb, filter_type, kernel)
            info_text = f"{filter_type} (Kernel {kernel}x{kernel})"

        with col_view:
            render_comparison(img_rgb, processed_img, info_text)

    # TAB 3: Edge Detection
    with tab3:
        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            edge_type = st.selectbox("Algoritma Tepi:", ["Canny Edge Detector", "Sobel Gradient", "Laplacian Operator"])
            
            if edge_type == "Canny Edge Detector":
                t1 = st.slider("Ambang Rendah (T1):", 0, 255, 80)
                t2 = st.slider("Ambang Tinggi (T2):", 0, 255, 180)
                processed_img = apply_edge_detection(img_rgb, "Canny", t1, t2)
                info_text = f"Canny (T1={t1}, T2={t2})"
            elif edge_type == "Sobel Gradient":
                processed_img = apply_edge_detection(img_rgb, "Sobel")
                info_text = "Sobel Gradient Magnitude"
            elif edge_type == "Laplacian Operator":
                processed_img = apply_edge_detection(img_rgb, "Laplacian")
                info_text = "Laplacian 2nd Derivative"

        with col_view:
            render_comparison(img_rgb, processed_img, info_text)

    # TAB 4: Color Spaces
    with tab4:
        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            space_choice = st.selectbox("Ruang Warna Sasaran:", ["HSV", "LAB", "YCrCb"])
            processed_img = convert_color_space(img_rgb, space_choice)
            info_text = f"Ruang Warna: {space_choice}"

        with col_view:
            render_comparison(img_rgb, processed_img, info_text)

    # TAB 5: Histogram
    with tab5:
        st.markdown('<div class="clean-metric-label" style="margin-bottom: 0.5rem;">Distribusi Nilai Intensitas Kanal Warna</div>', unsafe_allow_html=True)
        h_col1, h_col2 = st.columns([1, 1])
        with h_col1:
            st.image(img_rgb, caption="Citra Input Asli", use_container_width=True)
        with h_col2:
            fig = plot_histogram(img_rgb)
            st.pyplot(fig)


def render_comparison(source_img, result_img, info_text):
    """Menampilkan komparasi citra secara bersih dan rapi."""
    st.markdown(f'<div style="font-weight: 700; color: var(--text); margin-bottom: 0.5rem; font-size: 0.9rem;">Hasil Pemrosesan: <span style="color: var(--accent);">{info_text}</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.image(source_img, caption="Citra Asli", use_container_width=True)
        
    with c2:
        if result_img is not None:
            st.image(result_img, caption=f"Output: {info_text}", use_container_width=True, clamp=True)
            
            pil_res = Image.fromarray(result_img)
            buf = io.BytesIO()
            pil_res.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Unduh Hasil Citra (PNG)",
                data=buf.getvalue(),
                file_name="hasil_pcd.png",
                mime="image/png"
            )
