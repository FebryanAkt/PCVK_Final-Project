import streamlit as st
import importlib
from config import APP_TITLE, APP_SUBTITLE, VERSION, JOURNAL_TITLE
import pages_ui.home
import pages_ui.face_shape_page
import pages_ui.image_processing
import pages_ui.computer_vision
import pages_ui.webcam_test

# Pastikan modul selalu direload ke kode terbaru jika terjadi pembaruan berkas
importlib.reload(pages_ui.home)
importlib.reload(pages_ui.face_shape_page)
importlib.reload(pages_ui.image_processing)
importlib.reload(pages_ui.computer_vision)
importlib.reload(pages_ui.webcam_test)

from pages_ui.home import render_home_page
from pages_ui.face_shape_page import render_face_shape_page
from pages_ui.image_processing import render_image_processing_page
from pages_ui.computer_vision import render_cv_page
from pages_ui.webcam_test import render_webcam_page

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title=f"{APP_TITLE} | Face Shape & Vision Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Design System & Custom CSS
st.markdown("""
<style>
    /* Global Typography & Font Family */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero Banner */
    .vl-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #1D4ED8 100%);
        color: #F8FAFC;
        padding: 2.2rem 2.4rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .vl-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .vl-hero-desc {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        margin-bottom: 0;
        line-height: 1.5;
        max-width: 800px;
    }

    /* Cards & Containers */
    .vl-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
    }
    .vl-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.06);
    }

    /* Stat Cards */
    .vl-stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .vl-stat-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .vl-stat-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }

    /* Badges / Chips */
    .vl-badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 9999px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .vl-badge-success {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .vl-badge-danger {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .vl-badge-info {
        background-color: #DBEAFE;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .vl-badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }

    /* Custom Header Pills */
    .vl-section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        letter-spacing: -0.02em;
    }

    /* Buttons Styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        border: 1px solid #CBD5E1;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button {
        width: 100%;
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #64748B;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        font-weight: 700;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-brand-box {
        padding: 1rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigasi Utama
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">{APP_TITLE.upper()}</div>
        <div class="sidebar-brand-sub">Platform Pengolahan Citra dan Visi Komputer</div>
        <div style="margin-top: 0.6rem;">
            <span class="vl-badge vl-badge-info">v{VERSION}</span>
            <span class="vl-badge vl-badge-neutral">Jurnal Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Navigasi Modul**")
    selected_menu = st.radio(
        "Pilih Modul Navigasi:",
        [
            "Beranda dan Status",
            "Klasifikasi Bentuk Wajah (Jurnal)",
            "Pengolahan Citra (PCD)",
            "Computer Vision dan ML",
            "Uji Coba Webcam"
        ],
        label_visibility="collapsed",
        index=1  # Default langsung ke modul jurnal yang diminta!
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div class="vl-stat-box" style="margin-top: 1rem;">
        <div class="vl-stat-label">Referensi Jurnal</div>
        <div style="font-weight: 700; color: #0F172A; font-size: 0.85rem; line-height: 1.3;">{JOURNAL_TITLE}</div>
        <div style="font-size: 0.74rem; color: #64748B; margin-top: 0.35rem;">Mehta & Mahmoud &bull; Dlib 13 Landmarks &bull; 14 PCA Features</div>
    </div>
    """, unsafe_allow_html=True)

# 4. Halaman Router Berdasarkan Menu yang Dipilih
if selected_menu == "Beranda dan Status":
    render_home_page()
elif selected_menu == "Klasifikasi Bentuk Wajah (Jurnal)":
    render_face_shape_page()
elif selected_menu == "Pengolahan Citra (PCD)":
    render_image_processing_page()
elif selected_menu == "Computer Vision dan ML":
    render_cv_page()
elif selected_menu == "Uji Coba Webcam":
    render_webcam_page()

# 5. Footer Modern
st.markdown("""
<div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #E2E8F0; text-align: center; color: #94A3B8; font-size: 0.8rem;">
    VisionLab Studio &bull; Implementasi Human Face Shape Classification (Mehta & Mahmoud) &bull; 2026
</div>
""", unsafe_allow_html=True)
