import streamlit as st
from utils.bridges_catalog import BRIDGES_PRODUCTS

def render_catalog_page():
    # ── Editorial Header ──────────────────────────────────────
    st.markdown("""
    <div class="be-page-header">
        <div class="be-eyebrow">BRIDGES EYEWEAR · OFFICIAL COLLECTION</div>
        <h1 class="be-page-title">Koleksi Frame Optik &amp; Sunwear</h1>
        <p class="be-page-subtitle">
            Kacamata kontemporer berdesain minimalis internasional dengan standar kualitas 
            <strong>Optik Melawai Group</strong>. Pilihan material Full Acetate Italia, 
            Bridges+ Titanium Jepang (11g), hingga kombinasi retro-modern.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive Filter Bar ────────────────────────────────
    st.markdown('<div class="be-filter-bar">', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.5, 1.5, 1.5, 1.5])
    
    with f1:
        face_filter = st.selectbox(
            "Filter Bentuk Wajah:",
            ["Semua Bentuk Wajah", "Round (Bulat)", "Square (Kotak)", "Oval", "Heart (Hati)", "Oblong (Panjang)"],
            index=0
        )
    with f2:
        shape_filter = st.selectbox(
            "Siluet Frame:",
            ["Semua Bentuk Frame", "Square", "Round", "Wayfarer", "Cat Eye", "Clubmaster", "Geometric"],
            index=0
        )
    with f3:
        material_filter = st.selectbox(
            "Material:",
            ["Semua Material", "Acetate", "Titanium", "Metal / Combination", "TR-90 Polymer"],
            index=0
        )
    with f4:
        sort_by = st.selectbox(
            "Urutkan:",
            ["Paling Populer", "Harga: Rendah ke Tinggi", "Harga: Tinggi ke Rendah", "Bobot Paling Ringan"],
            index=0
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Filter Logic ──────────────────────────────────────────
    filtered = BRIDGES_PRODUCTS.copy()

    if face_filter != "Semua Bentuk Wajah":
        target_face = face_filter.split(" ")[0]
        filtered = [p for p in filtered if target_face in p["suitable_faces"]]

    if shape_filter != "Semua Bentuk Frame":
        filtered = [p for p in filtered if shape_filter.lower() in p["shape"].lower()]

    if material_filter != "Semua Material":
        filtered = [p for p in filtered if material_filter.lower() in p["material"].lower()]

    if sort_by == "Harga: Rendah ke Tinggi":
        filtered = sorted(filtered, key=lambda x: x["price_num"])
    elif sort_by == "Harga: Tinggi ke Rendah":
        filtered = sorted(filtered, key=lambda x: x["price_num"], reverse=True)
    elif sort_by == "Bobot Paling Ringan":
        filtered = sorted(filtered, key=lambda x: int(x["specs"]["weight"].split(" ")[0]))

    # ── Total Count ───────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin: 1.2rem 0; padding-bottom: 0.8rem; border-bottom: 1px solid var(--be-border);">
        <div style="font-size: 0.85rem; color: var(--be-text-muted); font-weight: 500;">
            Menampilkan <strong style="color: var(--be-text-primary);">{len(filtered)}</strong> model kacamata Bridges Eyewear
        </div>
        <div style="font-size: 0.78rem; color: var(--be-accent); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
            ✓ Termasuk Lensa Standar Optik Melawai (Index 1.6)
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not filtered:
        st.markdown("""
        <div class="be-empty-state">
            <div style="font-size: 2.2rem; margin-bottom: 0.6rem;"></div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--be-text-primary);">Tidak ada frame yang cocok dengan filter</div>
            <div style="font-size: 0.85rem; color: var(--be-text-muted); margin-top: 0.3rem;">Coba ubah opsi filter bentuk wajah atau material Anda.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Product Grid (2 columns or 3 columns) ─────────────────
    cols = st.columns(3)
    for idx, prod in enumerate(filtered):
        col_idx = idx % 3
        with cols[col_idx]:
            render_product_card(prod)


def render_product_card(prod: dict, is_recommended: bool = False):
    """Merender kartu produk kacamata ala Bridges Eyewear e-commerce."""
    badge_html = f'<span class="be-badge be-badge-gold">{prod["badge"]}</span>' if "badge" in prod else ""
    if is_recommended:
        badge_html = '<span class="be-badge be-badge-match">⭐ COCOK DENGAN WAJAH ANDA</span> ' + badge_html

    suitable_tags = " · ".join(prod["suitable_faces"])
    s = prod["specs"]

    st.markdown(f"""
    <div class="be-product-card">
        <div class="be-card-top">
            <div class="be-card-badge-row">{badge_html}</div>
            <span class="be-size-tag">{prod['size_tag']}</span>
        </div>
        
        <!-- Frame Visual Sizing Graphic -->
        <div class="be-frame-visual">
            <div class="be-frame-icon-box">
                <span class="be-frame-shape-icon"></span>
                <div class="be-frame-code">{prod['optical_code']}</div>
            </div>
            <div class="be-frame-dimension-pill">
                Lebar Total: <strong>{s['frame_width']} mm</strong> · Lensa: <strong>{s['lens_width']} mm</strong> · Bridge: <strong>{s['bridge']} mm</strong>
            </div>
        </div>

        <div class="be-card-body">
            <div class="be-collection-sub">{prod['collection'].upper()}</div>
            <h3 class="be-product-name">{prod['name']}</h3>
            <div class="be-product-color">{prod['color']} · {prod['material']}</div>
            
            <div class="be-suitable-box">
                <span class="be-suitable-label">Rekomendasi Wajah:</span>
                <span class="be-suitable-vals">{suitable_tags}</span>
            </div>

            <p class="be-product-desc">{prod['description']}</p>
            
            <div class="be-card-footer">
                <div class="be-price">{prod['price_idr']}</div>
                <div class="be-weight-tag">{s['weight']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
