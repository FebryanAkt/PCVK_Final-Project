import streamlit as st
import numpy as np
from PIL import Image
from utils.face_shape_analyzer import (
    analyze_face_shape_and_glasses,
    create_81_landmark_visualization,
    create_geometric_visualization,
    create_combined_visualization
)

def render_webcam_page():
    # ── Header Ringkas ─────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
            Uji Coba Kamera Langsung (Webcam)
        </h2>
        <div style="color: var(--text-muted); font-size: 0.88rem; line-height: 1.5;">
            Ambil tangkapan foto dari webcam untuk mengekstrak 81 landmark Dlib secara real-time, 
            menghitung jarak geometris wajah, dan menentukan rekomendasi kacamata yang cocok.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Konfigurasi Visualisasi Anotasi ───────────────────────
    c_vis, c_conf = st.columns([2, 1])
    with c_vis:
        vis_mode = st.selectbox(
            "Gaya Visualisasi Anotasi Wajah:",
            [
                "Kombinasi (81 Titik + Garis Jarak D1-D7)",
                "81 Titik Landmark & Angka Indeks (Kuning & Merah)",
                "Garis Jarak Geometris (D1 - D7)"
            ],
            index=0
        )
    with c_conf:
        min_conf = st.slider("Confidence Min:", 0.1, 1.0, 0.5, step=0.05)

    # ── Jendela Kamera & Pratinjau ─────────────────────────────
    st.markdown('<div class="section-title">Pengambilan Citra Kamera</div>', unsafe_allow_html=True)
    col_cam, col_res = st.columns(2)

    with col_cam:
        st.caption("Posisikan wajah tegak lurus menghadap kamera dengan pencahayaan cukup:")
        camera_photo = st.camera_input("Kamera", label_visibility="collapsed")

    annotated_default = None
    res = None

    with col_res:
        if camera_photo is not None:
            pil_img = Image.open(camera_photo)
            img_rgb = np.array(pil_img.convert("RGB"))

            with st.spinner("Mengekstrak landmark dan mengukur geometri wajah..."):
                annotated_default, res = analyze_face_shape_and_glasses(img_rgb, min_conf=min_conf)

            if annotated_default is None:
                st.warning(f"Keterangan sistem: {res}")
                st.image(img_rgb, caption="Citra Kamera Asli", use_container_width=True)
            else:
                if "Angka Indeks" in vis_mode:
                    display_img = create_81_landmark_visualization(img_rgb, res["points_81"])
                elif "Garis Jarak" in vis_mode:
                    display_img = create_geometric_visualization(img_rgb, res["points_81"])
                else:
                    display_img = create_combined_visualization(img_rgb, res["points_81"])

                st.image(display_img, caption="Hasil Anotasi Geometri Wajah", use_container_width=True)
                st.success("✓ Wajah berhasil terdeteksi dan dianalisis.")
        else:
            st.markdown("""
            <div class="clean-card" style="text-align: center; padding: 3rem 1.5rem; color: var(--text-muted);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📷</div>
                <div style="font-weight: 700; color: var(--text); font-size: 0.95rem; margin-bottom: 0.2rem;">
                    Menunggu Tangkapan Kamera
                </div>
                <div style="font-size: 0.82rem;">
                    Klik tombol <strong>Take Photo</strong> di sebelah kiri untuk memulai analisis bentuk wajah secara instan.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Hasil Detail Jika Snapshot Berhasil ────────────────────
    if camera_photo is not None and annotated_default is not None and res is not None and isinstance(res, dict):
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
