"""
Bridges Eyewear Product Catalog & Recommendation Engine
Data produk resmi dan kurasi frame kacamata Bridges Eyewear (Optik Melawai Group)
disesuaikan dengan bentuk wajah hasil ekstraksi landmark geometris.
"""

BRIDGES_PRODUCTS = [
    {
        "id": "BE-KAITO-01",
        "name": "Bridges - Kaito",
        "category": "Optical Glasses",
        "collection": "Acetate Icons",
        "shape": "Square",
        "shape_id": "Kotak / Square",
        "material": "Full Italian Acetate",
        "color": "Matte Charcoal Black",
        "price_idr": "Rp 850.000",
        "price_num": 850000,
        "size_tag": "Medium",
        "specs": {
            "frame_width": 138,
            "lens_width": 52,
            "bridge": 18,
            "temple": 145,
            "weight": "24 gram"
        },
        "optical_code": "52 [] 18 - 145",
        "suitable_faces": ["Round", "Oval"],
        "avoid_faces": ["Square"],
        "description": "Siluet persegi modern dengan ketebalan acetate seimbang. Memberikan kontur garis tegas dan efek visual ramping pada wajah membulat.",
        "badge": "BEST SELLER",
        "tag": "Acetate Series"
    },
    {
        "id": "BE-HARPER-02",
        "name": "Bridges - Harper",
        "category": "Optical Glasses",
        "collection": "The Timeless",
        "shape": "Round",
        "shape_id": "Bulat / Round",
        "material": "Mixed Acetate & Gold Metal Rim",
        "color": "Dark Tortoise & Brushed Gold",
        "price_idr": "Rp 950.000",
        "price_num": 950000,
        "size_tag": "Small to Medium",
        "specs": {
            "frame_width": 133,
            "lens_width": 49,
            "bridge": 20,
            "temple": 142,
            "weight": "20 gram"
        },
        "optical_code": "49 [] 20 - 142",
        "suitable_faces": ["Square", "Oblong", "Heart"],
        "avoid_faces": ["Round"],
        "description": "Desain panto klasik dengan jembatan hidung keyhole berpadu metal gold elegan. Kurvanya melembutkan rahang kotak dan memperpendek proporsi wajah panjang.",
        "badge": "STAFF PICK",
        "tag": "Combination Frame"
    },
    {
        "id": "BE-ARLO-TITANIUM",
        "name": "Bridges+ Arlo Titanium",
        "category": "Optical Glasses",
        "collection": "Bridges+ Premium Line",
        "shape": "Round",
        "shape_id": "Bulat Oval / Round",
        "material": "Pure Japanese Titanium (Hypoallergenic)",
        "color": "Brushed Silver Platinum",
        "price_idr": "Rp 1.250.000",
        "price_num": 1250000,
        "size_tag": "Medium",
        "specs": {
            "frame_width": 134,
            "lens_width": 48,
            "bridge": 19,
            "temple": 140,
            "weight": "11 gram (Ultra-Light)"
        },
        "optical_code": "48 [] 19 - 140",
        "suitable_faces": ["Square", "Heart", "Oval"],
        "avoid_faces": ["Round"],
        "description": "Lini flagship Bridges+ bermaterial titanium murni berbobot ultra-ringan 11g. Tahan korosi, fleksibel, serta tidak meninggalkan bekas merah di hidung.",
        "badge": "PREMIUM TITANIUM",
        "tag": "Bridges+ 11g"
    },
    {
        "id": "BE-CHLOE-04",
        "name": "Bridges - Chloe",
        "category": "Optical Glasses",
        "collection": "Women's Capsule",
        "shape": "Cat Eye",
        "shape_id": "Cat-Eye Melengkung",
        "material": "Handcrafted Crystal Acetate",
        "color": "Crystal Champagne Rose",
        "price_idr": "Rp 850.000",
        "price_num": 850000,
        "size_tag": "Medium",
        "specs": {
            "frame_width": 136,
            "lens_width": 51,
            "bridge": 17,
            "temple": 142,
            "weight": "22 gram"
        },
        "optical_code": "51 [] 17 - 142",
        "suitable_faces": ["Oval", "Square", "Round"],
        "avoid_faces": ["Heart"],
        "description": "Aksen sudut cat-eye bersahaja yang mengangkat garis tulang pipi. Warna crystal champagne transparan memberi kesan segar, modis, dan mewah.",
        "badge": "TRENDING",
        "tag": "Crystal Acetate"
    },
    {
        "id": "BE-OLIVER-05",
        "name": "Bridges - Oliver",
        "category": "Optical Glasses",
        "collection": "The Icons",
        "shape": "Wayfarer",
        "shape_id": "Wayfarer Retro",
        "material": "Premium Thick Acetate",
        "color": "Vintage Amber Havana",
        "price_idr": "Rp 850.000",
        "price_num": 850000,
        "size_tag": "Medium to Wide",
        "specs": {
            "frame_width": 139,
            "lens_width": 50,
            "bridge": 20,
            "temple": 145,
            "weight": "26 gram"
        },
        "optical_code": "50 [] 20 - 145",
        "suitable_faces": ["Round", "Oval", "Heart"],
        "avoid_faces": ["Square"],
        "description": "Siluet wayfarer retro abadi dengan proporsi jembatan hidung lebar. Sangat cocok menyeimbangkan pipi penuh pada wajah bulat maupun dahi wajah hati.",
        "badge": "CLASSIC ICON",
        "tag": "Handcrafted"
    },
    {
        "id": "BE-VANCE-TITANIUM",
        "name": "Bridges+ Vance",
        "category": "Optical Glasses",
        "collection": "Bridges+ Premium Line",
        "shape": "Geometric",
        "shape_id": "Geometris / Hexagonal",
        "material": "Beta-Titanium & Monel",
        "color": "Matte Gunmetal Grey",
        "price_idr": "Rp 1.350.000",
        "price_num": 1350000,
        "size_tag": "Medium",
        "specs": {
            "frame_width": 137,
            "lens_width": 50,
            "bridge": 19,
            "temple": 145,
            "weight": "13 gram"
        },
        "optical_code": "50 [] 19 - 145",
        "suitable_faces": ["Round", "Oval"],
        "avoid_faces": ["Square"],
        "description": "Frame bersudut segi-enam (hexagonal) halus berbahan beta-titanium elastis. Menghadirkan karakter arsitektural intelektual yang kontemporer.",
        "badge": "NEW ARRIVAL",
        "tag": "Beta-Titanium"
    },
    {
        "id": "BE-DYLAN-07",
        "name": "Bridges - Dylan",
        "category": "Optical Glasses",
        "collection": "The Browline",
        "shape": "Clubmaster",
        "shape_id": "Clubmaster / Semi-Rimless",
        "material": "Acetate Brow + Stainless Steel Lower",
        "color": "Glossy Onyx Black & Gold",
        "price_idr": "Rp 950.000",
        "price_num": 950000,
        "size_tag": "Medium",
        "specs": {
            "frame_width": 138,
            "lens_width": 51,
            "bridge": 19,
            "temple": 145,
            "weight": "23 gram"
        },
        "optical_code": "51 [] 19 - 145",
        "suitable_faces": ["Oval", "Heart", "Round"],
        "avoid_faces": ["Square"],
        "description": "Desain semi-rimless legendaris bergaya intelektual era 60-an. Aksen alis atas yang tegas memberi struktur wibawa pada area mata.",
        "badge": "SIGNATURE",
        "tag": "Combination"
    },
    {
        "id": "BE-LUNA-08",
        "name": "Bridges - Luna",
        "category": "Optical Glasses",
        "collection": "Minimalist Metal",
        "shape": "Oversized Round",
        "shape_id": "Bulat Oversized / Deep Frame",
        "material": "Ultra-Fine Stainless Steel",
        "color": "Metallic Rose Gold",
        "price_idr": "Rp 950.000",
        "price_num": 950000,
        "size_tag": "Wide / Deep",
        "specs": {
            "frame_width": 142,
            "lens_width": 53,
            "bridge": 18,
            "temple": 142,
            "weight": "18 gram"
        },
        "optical_code": "53 [] 18 - 142",
        "suitable_faces": ["Oblong", "Square"],
        "avoid_faces": ["Round"],
        "description": "Lensa bulat tinggi (tall lens) oversized yang sengaja dirancang untuk memotong dimensi vertikal wajah oblong agar tampak lebih seimbang dan proporsional.",
        "badge": "BEST FOR OBLONG",
        "tag": "Oversized Slim"
    },
    {
        "id": "BE-FELIX-ACTIVE",
        "name": "Bridges Active - Felix",
        "category": "Optical & Screenspecs",
        "collection": "Bridges Active",
        "shape": "Rectangle",
        "shape_id": "Persegi Panjang Sport",
        "material": "Swiss TR-90 Memory Polymer + Rubber Temple",
        "color": "Matte Navy Stealth",
        "price_idr": "Rp 800.000",
        "price_num": 800000,
        "size_tag": "Medium to Wide",
        "specs": {
            "frame_width": 140,
            "lens_width": 54,
            "bridge": 17,
            "temple": 142,
            "weight": "16 gram"
        },
        "optical_code": "54 [] 17 - 142",
        "suitable_faces": ["Round", "Oval"],
        "avoid_faces": ["Square", "Oblong"],
        "description": "Diciptakan untuk mobilitas tinggi dan olahraga. Material TR-90 berdaya lentur ekstrem dilengkapi nosepad silikon antiselip meski berkeringat.",
        "badge": "ACTIVE TECH",
        "tag": "TR-90 Polymer"
    },
    {
        "id": "BE-SUNEATER-MIRAGE",
        "name": "Bridges x Sun Eater - Mirage",
        "category": "Sunglasses / Sunwear",
        "collection": "Limited Collab Capsule",
        "shape": "Chunky Square",
        "shape_id": "Kotak Tebal / Bold Square",
        "material": "8mm High-Density Block Acetate",
        "color": "Deep Pitch Black (Lens UV400 Dark Grey)",
        "price_idr": "Rp 1.100.000",
        "price_num": 1100000,
        "size_tag": "Wide",
        "specs": {
            "frame_width": 144,
            "lens_width": 53,
            "bridge": 20,
            "temple": 148,
            "weight": "34 gram"
        },
        "optical_code": "53 [] 20 - 148",
        "suitable_faces": ["Round", "Oval"],
        "avoid_faces": ["Square"],
        "description": "Kolaborasi spesial dengan label musik indie Sun Eater. Siluet kotak chunky 8mm tebal bergaya streetwear modern dengan proteksi UV400 100%.",
        "badge": "LIMITED EDITION",
        "tag": "Collaboration"
    }
]

def get_recommendations_for_face_shape(face_shape: str):
    """Mengembalikan daftar frame Bridges Eyewear yang cocok dengan bentuk wajah tertentu."""
    matched = []
    other = []
    target = face_shape.capitalize() if face_shape else "Oval"
    for prod in BRIDGES_PRODUCTS:
        if target in prod["suitable_faces"]:
            matched.append(prod)
        else:
            other.append(prod)
    return matched, other
