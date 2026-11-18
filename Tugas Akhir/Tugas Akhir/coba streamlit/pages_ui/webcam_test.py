import streamlit as st
import numpy as np
from PIL import Image
from utils.cv_helpers import to_grayscale, apply_edge_detection, apply_threshold
from utils.model_loader import VisionModelManager
from utils.face_shape_analyzer import (
    analyze_face_shape_and_glasses,
    create_81_landmark_visualization,
    create_geometric_visualization,
    create_combined_visualization
)

def render_webcam_page():
    # Header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <span class="vl-badge vl-badge-info" style="margin-bottom: 0.5rem;">Uji Coba Kamera Interaktif</span>
        <h2 style="font-weight: 800; color: #0F172A; margin-top: 0.2rem; margin-bottom: 0.2rem; letter-spacing: -0.02em;">
            Uji Coba Webcam: Ekstraksi 81 Landmark Wajah dan Rekomendasi Ukuran Kacamata
        </h2>
        <div style="color: #64748B; font-size: 0.92rem; line-height: 1.5;">
            Ambil tangkapan foto langsung dari kamera laptop atau PC untuk mengekstrak 81 titik landmark (codeniko Dlib-81), 
            menghitung parameter geometris (lebar pelipis, tulang pipi, rahang, jembatan hidung), dan mengestimasi ukuran kacamata yang cocok secara instan.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Mode Pipeline & Parameter
    st.sidebar.markdown("**Konfigurasi Pemrosesan Webcam**")
    pipeline_mode = st.sidebar.selectbox(
        "Pilih Mode Pipeline:",
        [
            "81 Landmark Wajah & Ukuran Kacamata (Jurnal)",
            "Deteksi Tepi (Canny Edge)",
            "Binarisasi Ambang (Thresholding)",
            "MediaPipe Hand Landmarker"
        ],
        index=0
    )

    vis_mode = "Kombinasi (81 Titik + Garis Jarak D1-D7)"
    min_conf = 0.5
    if pipeline_mode == "81 Landmark Wajah & Ukuran Kacamata (Jurnal)":
        vis_mode = st.sidebar.selectbox(
            "Gaya Visualisasi Anotasi Wajah:",
            [
                "81 Titik Landmark & Angka Indeks (Kuning & Merah)",
                "Garis Jarak Geometris (D1 - D7)",
                "Kombinasi (81 Titik + Garis Jarak D1-D7)"
            ],
            index=0
        )
        min_conf = st.sidebar.slider("Confidence Threshold:", 0.1, 1.0, 0.5, step=0.05)

    # Layout Kamera dan Hasil Snapshot
    col_cam, col_res = st.columns(2)

    with col_cam:
        st.markdown('<div class="vl-stat-label">Input Kamera Webcam Langsung</div>', unsafe_allow_html=True)
        st.caption("Posisikan wajah Anda tegak lurus menghadap kamera, lalu klik 'Take Photo':")
        camera_photo = st.camera_input("Pratinjau Kamera", label_visibility="collapsed")

    with col_res:
        st.markdown('<div class="vl-stat-label">Pratinjau Hasil Anotasi Snapshot</div>', unsafe_allow_html=True)
        
        if camera_photo is not None:
            pil_img = Image.open(camera_photo)
            img_rgb = np.array(pil_img.convert("RGB"))

            # JIKA MODE 81 LANDMARK & KACAMATA
            if pipeline_mode == "81 Landmark Wajah & Ukuran Kacamata (Jurnal)":
                with st.spinner("Mengekstrak 81 titik koordinat dan menghitung ukuran kacamata..."):
                    annotated_default, res = analyze_face_shape_and_glasses(img_rgb, min_conf=min_conf)

                if annotated_default is None:
                    st.warning(f"Keterangan: {res}")
                    st.image(img_rgb, caption="Citra Tangkapan Kamera Asli", use_container_width=True)
                else:
                    # Pilih gaya gambar sesuai preferensi
                    if vis_mode == "81 Titik Landmark & Angka Indeks (Kuning & Merah)":
                        display_img = create_81_landmark_visualization(img_rgb, res["points_81"])
                    elif vis_mode == "Garis Jarak Geometris (D1 - D7)":
                        display_img = create_geometric_visualization(img_rgb, res["points_81"])
                    else:
                        display_img = create_combined_visualization(img_rgb, res["points_81"])

                    st.image(display_img, caption=f"Anotasi Kamera: {vis_mode}", use_container_width=True)
                    st.success("Wajah berhasil terdeteksi dan 81 koordinat telah dihitung.")

            # MODE LAINNYA
            elif pipeline_mode == "Deteksi Tepi (Canny Edge)":
                edges = apply_edge_detection(img_rgb, "Canny", 100, 200)
                st.image(edges, caption="Hasil Penapisan Tepi Canny", use_container_width=True)

            elif pipeline_mode == "Binarisasi Ambang (Thresholding)":
                thresh_val = st.sidebar.slider("Nilai Ambang:", 0, 255, 128)
                bin_img = apply_threshold(img_rgb, "Binary", thresh_val)
                st.image(bin_img, caption=f"Hasil Binarisasi (T={thresh_val})", use_container_width=True)

            elif pipeline_mode == "MediaPipe Hand Landmarker":
                if VisionModelManager.is_hand_model_available():
                    hand_res, h_info = VisionModelManager.detect_hand_landmarks(img_rgb)
                    st.image(hand_res, caption=f"Hand Landmarker ({h_info.get('hands_detected', 0)} Tangan)", use_container_width=True)
                else:
                    st.error("Model hand_landmarker.task tidak ditemukan.")

        else:
            st.markdown("""
            <div style="background: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 3.5rem 1.5rem; text-align: center; color: #64748B;">
                <div style="font-weight: 700; color: #475569; font-size: 0.95rem; margin-bottom: 0.3rem;">Menunggu Pengambilan Frame Kamera</div>
                <div style="font-size: 0.82rem;">Klik tombol Take Photo pada jendela kamera di sebelah kiri untuk memulai analisis geometri dan rekomendasi kacamata.</div>
            </div>
            """, unsafe_allow_html=True)

    # TAMPILKAN HASIL DETAIL BILA SUDAH MENGAMBIL FOTO DENGAN MODE 81 LANDMARK
    if camera_photo is not None and pipeline_mode == "81 Landmark Wajah & Ukuran Kacamata (Jurnal)" and annotated_default is not None:
        st.markdown("---")
        gl = res["glasses"]
        pred_shape = res["predicted_shape"]
        prob = res["probabilities"].get(pred_shape, 0.0)

        # 1. BANNER UTAMA REKOMENDASI KACAMATA
        st.markdown(f"""
        <div class="vl-hero" style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 65%, #2563EB 100%); padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;">
            <span class="vl-badge vl-badge-info" style="background: rgba(255, 255, 255, 0.15); color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.25); margin-bottom: 0.6rem;">
                Hasil Analisis Tangkapan Webcam Langsung
            </span>
            <div style="font-size: 2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                UKURAN FRAME KACAMATA: {gl['size_category']}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 0.8rem; align-items: baseline;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #93C5FD;">
                    Kode Standar Optik: <code>{gl['optical_code']}</code>
                </div>
                <div style="font-size: 0.95rem; color: #E2E8F0;">
                    Bentuk Wajah Anda: <strong>{pred_shape.upper()} ({prob}%)</strong>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.5rem;">
                Ukuran dihitung secara antropometrik dari rasio Lebar Pelipis (D2), Lebar Tulang Pipi (D1), dan Lebar Jembatan Hidung (titik 39-42).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. 4 KARTU METRIK DIMENSI KACAMATA
        st.markdown('<div class="vl-section-header">Spesifikasi Ukuran Bingkai Kacamata Rekomendasi</div>', unsafe_allow_html=True)
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
                <div class="vl-stat-label">Panjang Tangkai (Temple)</div>
                <div class="vl-stat-val" style="color: #2563EB;">{gl['temple_arm_length']}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">Temple Arm Length</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. MODEL BINGKAI YANG COCOK VS PERLU DIHINDARI
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

        # 4. TABEL PENGUKURAN JARAK ANATOMIS WAJAH
        st.markdown('<div class="vl-section-header">Hasil Pengukuran Jarak Geometris Wajah Anda (Rumus Jurnal)</div>', unsafe_allow_html=True)
        px = res["measurements_px"]
        mm = res["measurements_mm"]
        r = res["ratios"]

        meas_data = [
            {
                "Parameter Pengukuran": "Lebar Pelipis / Dahi (Temple Width)",
                "Titik Landmark": "Titik 75 ke 79 (D2)",
                "Nilai Piksel": f"{px['D2']:.1f} px",
                "Estimasi Milimeter": f"{mm['Temple_Width_mm']} mm"
            },
            {
                "Parameter Pengukuran": "Lebar Pipi / Tulang Pipi (Cheekbone Width)",
                "Titik Landmark": "Titik 2 ke 14 (D1)",
                "Nilai Piksel": f"{px['D1']:.1f} px",
                "Estimasi Milimeter": f"{mm['Cheekbone_Width_mm']} mm"
            },
            {
                "Parameter Pengukuran": "Panjang Vertikal Wajah (Face Length)",
                "Titik Landmark": "Midpoint 69 & 72 ke Titik 8 (D3)",
                "Nilai Piksel": f"{px['D3']:.1f} px",
                "Estimasi Milimeter": f"{mm['Face_Length_mm']} mm"
            },
            {
                "Parameter Pengukuran": "Lebar Sudut Rahang (Jaw Width)",
                "Titik Landmark": "Titik 4 ke 12 (D5)",
                "Nilai Piksel": f"{px['D5']:.1f} px",
                "Estimasi Milimeter": f"{mm['Jaw_Width_mm']} mm"
            },
            {
                "Parameter Pengukuran": "Lebar Rahang Bawah (Lower Jaw Width)",
                "Titik Landmark": "Titik 6 ke 10 (D6)",
                "Nilai Piksel": f"{px['D6']:.1f} px",
                "Estimasi Milimeter": f"{round(px['D6'] * (mm['Cheekbone_Width_mm']/px['D1']), 1)} mm"
            },
            {
                "Parameter Pengukuran": "Lebar Dasar Dagu (Chin Base Width)",
                "Titik Landmark": "Titik 7 ke 9 (D7)",
                "Nilai Piksel": f"{px['D7']:.1f} px",
                "Estimasi Milimeter": f"{round(px['D7'] * (mm['Cheekbone_Width_mm']/px['D1']), 1)} mm"
            },
            {
                "Parameter Pengukuran": "Lebar Jembatan Hidung (Nose Bridge)",
                "Titik Landmark": "Titik 39 ke 42",
                "Nilai Piksel": f"{px['Bridge_Width']:.1f} px",
                "Estimasi Milimeter": f"{mm['Bridge_Width_mm']} mm"
            }
        ]
        st.dataframe(meas_data, use_container_width=True, hide_index=True)

        # 5. TABEL LENGKAP 81 KOORDINAT DARI WEBCAM
        with st.expander("Tabel Seluruh 81 Titik Koordinat Landmark Wajah dari Kamera (X, Y)"):
            st.caption("Koordinat piksel hasil ekstraksi wajah Anda dari titik 0 hingga 80:")
            st.dataframe(res["points_table"], use_container_width=True, hide_index=True)
