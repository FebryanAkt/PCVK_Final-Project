import streamlit as st
import importlib
from config import APP_TITLE, JOURNAL_TITLE, JOURNAL_AUTHORS

import pages_ui.home
import pages_ui.face_shape_page
import pages_ui.webcam_test

importlib.reload(pages_ui.home)
importlib.reload(pages_ui.face_shape_page)
importlib.reload(pages_ui.webcam_test)

from pages_ui.home import render_home_page
from pages_ui.face_shape_page import render_face_shape_page
from pages_ui.webcam_test import render_webcam_page

st.set_page_config(
    page_title="Face Shape AI & Vision Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
#  STATEFUL THEME MANAGEMENT (LIGHT / DARK TOGGLE)
# ═══════════════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "light"

is_dark = (st.session_state.theme == "dark")

# Definisi CSS Tokens & Overrides Berdasarkan State Theme
if is_dark:
    theme_css = """
    :root {
        --bg:             #0B0F19;
        --bg-subtle:      #111827;
        --surface:        #162032;
        --surface-card:   #162032;
        --border:         #24324D;
        --border-hover:   #3B4E72;
        --text:           #F8FAFC;
        --text-muted:     #CBD5E1;
        --text-light:     #94A3B8;
        --primary:        #F8FAFC;
        --primary-light:  #1E293B;
        --accent:         #38BDF8;
        --accent-soft:    #0C2640;
        --accent-text:    #38BDF8;
        --success:        #4ADE80;
        --success-soft:   #062E1A;
        --input-bg:       #162032;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
    }
    h1, h2, h3, h4, h5, h6,
    p, span, label, div[data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] span,
    [data-testid="stRadio"] p,
    [data-testid="stSelectbox"] *,
    [data-testid="stSlider"] *,
    [data-testid="stExpander"] summary *,
    [data-testid="stFileUploader"] *,
    .stButton > button,
    .stButton > button * {
        color: #F8FAFC !important;
    }
    .app-subtitle, .clean-metric-desc, .section-title, .simple-footer,
    [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {
        color: #94A3B8 !important;
    }
    .clean-metric-label {
        color: #CBD5E1 !important;
    }
    .stButton > button {
        background-color: #162032 !important;
        border-color: #24324D !important;
        color: #F8FAFC !important;
    }
    .stButton > button:hover {
        background-color: #1F2D44 !important;
        border-color: #3B4E72 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #162032 !important;
        border-color: #24324D !important;
        color: #F8FAFC !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="popover"] li {
        background-color: #162032 !important;
        color: #F8FAFC !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: #1F2D44 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #111827 !important;
        border-color: #24324D !important;
    }
    """
else:
    theme_css = """
    :root {
        --bg:             #FFFFFF;
        --bg-subtle:      #F8FAFC;
        --surface:        #FFFFFF;
        --surface-card:   #FFFFFF;
        --border:         #E2E8F0;
        --border-hover:   #CBD5E1;
        --text:           #0F172A;
        --text-muted:     #334155;
        --text-light:     #64748B;
        --primary:        #0F172A;
        --primary-light:  #F1F5F9;
        --accent:         #2563EB;
        --accent-soft:    #EFF6FF;
        --accent-text:    #1D4ED8;
        --success:        #15803D;
        --success-soft:   #F0FDF4;
        --input-bg:       #FFFFFF;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    h1, h2, h3, h4, h5, h6,
    p, span, label, div[data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] span,
    [data-testid="stRadio"] p,
    [data-testid="stSelectbox"] *,
    [data-testid="stSlider"] *,
    [data-testid="stExpander"] summary *,
    [data-testid="stFileUploader"] *,
    .stButton > button,
    .stButton > button * {
        color: #0F172A !important;
    }
    .app-subtitle, .clean-metric-desc, .section-title, .simple-footer,
    [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {
        color: #475569 !important;
    }
    .clean-metric-label {
        color: #334155 !important;
    }
    .stButton > button {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    .stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    [data-baseweb="popover"] li:hover {
        background-color: #F1F5F9 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

{theme_css}

:root {{
    --radius-sm:      6px;
    --radius:         10px;
    --font:           'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
}}

html, body, [class*="css"] {{
    font-family: var(--font) !important;
}}

.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1160px !important;
}}

/* Hide streamlit default elements */
[data-testid="collapsedControl"], section[data-testid="stSidebar"],
header[data-testid="stHeader"], #MainMenu, footer {{
    display: none !important;
}}

/* ── Minimalist Top Header ── */
.app-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}}
.app-title {{
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text) !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.app-subtitle {{
    font-size: 0.78rem;
    margin-top: 0.15rem;
}}
.app-badge {{
    display: inline-block;
    padding: 0.25rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 600;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 9999px;
    color: var(--text-muted) !important;
}}

/* ── Clean Section Cards ── */
.clean-card {{
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.15s ease;
}}
.clean-card:hover {{
    border-color: var(--border-hover);
}}

/* ── Metric Display ── */
.clean-metric {{
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
}}
.clean-metric-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}}
.clean-metric-val {{
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text) !important;
    line-height: 1.1;
}}
.clean-metric-desc {{
    font-size: 0.72rem;
    margin-top: 0.25rem;
}}

/* ── Section Title Divider ── */
.section-title {{
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.6rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}}

/* ── Confidence Box ── */
.conf-box {{
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.85rem 0.7rem;
    text-align: center;
}}
.conf-box.active {{
    border-color: var(--accent);
    background: var(--accent-soft);
}}
.conf-name {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted) !important;
}}
.conf-box.active .conf-name {{ color: var(--accent) !important; }}
.conf-val {{
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text) !important;
    margin: 0.2rem 0;
}}
.conf-box.active .conf-val {{ color: var(--accent) !important; }}

/* ── Streamlit Form Controls Refinement ── */
.stButton > button {{
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s ease !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    font-family: var(--font) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.5rem 1rem !important;
}}
.stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    font-weight: 700 !important;
}}

.stExpander {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--surface-card) !important;
}}
.stDataFrame {{
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
}}

/* ── Simple Footer ── */
.simple-footer {{
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 0.76rem;
}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MINIMALIST TOP HEADER WITH THEME TOGGLE BUTTON
# ═══════════════════════════════════════════════════════════════
c_head, c_theme = st.columns([4.2, 1.2])

with c_head:
    st.markdown(f"""
    <div>
        <div class="app-title"> Face Shape &amp; Vision Studio</div>
        <div class="app-subtitle">Implementasi Klasifikasi Bentuk Wajah Berbasis Landmark Geometris (Mehta &amp; Mahmoud)</div>
    </div>
    """, unsafe_allow_html=True)

with c_theme:
    theme_btn_label = "☀️ Mode Terang" if is_dark else "🌙 Mode Gelap"
    if st.button(theme_btn_label, key="theme_toggle_btn", use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

st.markdown("<div style='height: 0.6rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  NAVIGASI TABS
# ═══════════════════════════════════════════════════════════════
NAV_PAGES = [
    (" Klasifikasi Wajah (Jurnal)", "face"),
    (" Uji Coba Webcam", "webcam"),
    (" Ringkasan Metodologi", "home")
]

if st.session_state.get("active_nav") not in ["face", "webcam", "home"]:
    st.session_state.active_nav = "face"

nav_cols = st.columns(len(NAV_PAGES))
for idx, (label, key) in enumerate(NAV_PAGES):
    with nav_cols[idx]:
        is_active = (st.session_state.active_nav == key)
        display_label = f"● {label}" if is_active else label
        if st.button(display_label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_nav = key
            st.rerun()

st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ═══════════════════════════════════════════════════════════════
current = st.session_state.active_nav

if current == "face":
    render_face_shape_page()
elif current == "webcam":
    render_webcam_page()
elif current == "home":
    render_home_page()
else:
    render_face_shape_page()

# ═══════════════════════════════════════════════════════════════
#  CLEAN FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="simple-footer">
    Face Shape Classification with Machine Learning &bull; Referensi: {JOURNAL_TITLE} ({JOURNAL_AUTHORS}) &bull; Final Project PCD &amp; CV 2026
</div>
""", unsafe_allow_html=True)
