import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import FACE_LANDMARKER_PATH

# ==============================================================================
# PEMETAAN 81 TITIK LANDMARK (CODENIKO SHAPE PREDICTOR 81) KE MEDIAPIPE FACE MESH
# Berdasarkan: https://github.com/codeniko/shape_predictor_81_face_landmarks
# ==============================================================================

# Mapping lengkap 81 titik Dlib:
# 0-16: Kontur Garis Rahang (Jawline)
# 17-21: Alis Kanan (Right Eyebrow)
# 22-26: Alis Kiri (Left Eyebrow)
# 27-30: Batang Hidung (Nose Bridge)
# 31-35: Ujung Hidung / Lubang Hidung (Nose Base)
# 36-41: Mata Kanan (Right Eye)
# 42-47: Mata Kiri (Left Eye)
# 48-59: Bibir Luar (Outer Lips)
# 60-67: Bibir Dalam (Inner Lips)
# 68-80: 13 Titik Tambahan Area Dahi dan Garis Rambut (Forehead & Hairline)

DLIB_81_MAP = {
    # Kontur Rahang (0 - 16)
    0: (234, "Telinga/Pipi Kanan Atas"),
    1: (93, "Rahang Kanan 1"),
    2: (132, "Pipi/Tulang Pipi Kanan (D1)"),
    3: (58, "Rahang Kanan 3"),
    4: (172, "Sudut Rahang Kanan/Gonion (D5)"),
    5: (150, "Rahang Bawah Kanan 1"),
    6: (148, "Rahang Bawah Kanan/Mulut (D6)"),
    7: (149, "Dasar Dagu Kanan (D7)"),
    8: (152, "Puncak Dagu/Ujung Dagu (D3/D4)"),
    9: (378, "Dasar Dagu Kiri (D7)"),
    10: (377, "Rahang Bawah Kiri/Mulut (D6)"),
    11: (379, "Rahang Bawah Kiri 1"),
    12: (397, "Sudut Rahang Kiri/Gonion (D4/D5)"),
    13: (288, "Rahang Kiri 3"),
    14: (361, "Pipi/Tulang Pipi Kiri (D1)"),
    15: (323, "Rahang Kiri 1"),
    16: (454, "Telinga/Pipi Kiri Atas"),
    
    # Alis Kanan (17 - 21)
    17: (70, "Ujung Luar Alis Kanan"),
    18: (63, "Alis Kanan 2"),
    19: (105, "Puncak Alis Kanan"),
    20: (66, "Alis Kanan 4"),
    21: (107, "Ujung Dalam Alis Kanan"),
    
    # Alis Kiri (22 - 26)
    22: (336, "Ujung Dalam Alis Kiri"),
    23: (296, "Alis Kiri 2"),
    24: (334, "Puncak Alis Kiri"),
    25: (293, "Alis Kiri 4"),
    26: (300, "Ujung Luar Alis Kiri"),
    
    # Hidung (27 - 35)
    27: (168, "Pangkal Hidung Atas"),
    28: (6, "Batang Hidung 2"),
    29: (197, "Batang Hidung 3"),
    30: (4, "Puncak Ujung Hidung"),
    31: (98, "Cuping Hidung Kanan"),
    32: (97, "Lubang Hidung Kanan"),
    33: (2, "Dasar Septum Hidung"),
    34: (326, "Lubang Hidung Kiri"),
    35: (327, "Cuping Hidung Kiri"),
    
    # Mata Kanan (36 - 41)
    36: (33, "Sudut Luar Mata Kanan"),
    37: (160, "Kelopak Atas Mata Kanan 1"),
    38: (158, "Kelopak Atas Mata Kanan 2"),
    39: (133, "Sudut Dalam Mata Kanan"),
    40: (153, "Kelopak Bawah Mata Kanan 2"),
    41: (144, "Kelopak Bawah Mata Kanan 1"),
    
    # Mata Kiri (42 - 47)
    42: (362, "Sudut Dalam Mata Kiri"),
    43: (385, "Kelopak Atas Mata Kiri 1"),
    44: (387, "Kelopak Atas Mata Kiri 2"),
    45: (263, "Sudut Luar Mata Kiri"),
    46: (373, "Kelopak Bawah Mata Kiri 2"),
    47: (380, "Kelopak Bawah Mata Kiri 1"),
    
    # Bibir Luar (48 - 59)
    48: (61, "Sudut Bibir Kanan"),
    49: (185, "Bibir Atas Kanan 1"),
    50: (40, "Bibir Atas Kanan 2"),
    51: (39, "Puncak Cupid Bibir Atas"),
    52: (37, "Bibir Atas Kiri 2"),
    53: (0, "Bibir Atas Kiri 1"),
    54: (267, "Sudut Bibir Kiri"),
    55: (269, "Bibir Bawah Kiri 1"),
    56: (270, "Bibir Bawah Kiri 2"),
    57: (409, "Dasar Bibir Bawah Tengah"),
    58: (291, "Bibir Bawah Kanan 2"),
    59: (146, "Bibir Bawah Kanan 1"),
    
    # Bibir Dalam (60 - 67)
    60: (78, "Sudut Dalam Bibir Kanan"),
    61: (95, "Bibir Dalam Atas 1"),
    62: (88, "Bibir Dalam Atas Tengah"),
    63: (178, "Bibir Dalam Atas 2"),
    64: (87, "Sudut Dalam Bibir Kiri"),
    65: (14, "Bibir Dalam Bawah 2"),
    66: (317, "Bibir Dalam Bawah Tengah"),
    67: (402, "Bibir Dalam Bawah 1"),
    
    # 13 Titik Dahi Tambahan (Codeniko 81: 68 - 80)
    68: (108, "Dahi Kanan Luar Atas Alis"),
    69: (67, "Garis Rambut Dahi Kanan (D3)"),
    70: (109, "Dahi Kanan Tengah"),
    71: (10, "Puncak Garis Rambut Tengah"),
    72: (297, "Garis Rambut Dahi Kiri (D3)"),
    73: (338, "Dahi Kiri Tengah"),
    74: (299, "Dahi Kiri Luar Atas Alis"),
    75: (103, "Pelipis / Dahi Kanan (D2)"),
    76: (68, "Pelipis Kanan Bawah"),
    77: (104, "Pelipis Kanan Dekat Pipi"),
    78: (333, "Pelipis Kiri Dekat Pipi"),
    79: (332, "Pelipis / Dahi Kiri (D2)"),
    80: (337, "Pelipis Kiri Atas")
}

def calculate_angle(v1, v2):
    """Menghitung sudut antara dua vektor dalam derajat."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def analyze_face_shape_and_glasses(image_rgb: np.ndarray, min_conf: float = 0.5):
    """
    Mengekstrak 81 koordinat landmark wajah, menghitung seluruh parameter
    jarak anatomis (lebar pelipis, pipi, rahang, dagu, dll), rasio jurnal,
    serta menghasilkan rekomendasi ukuran dan bentuk kacamata yang cocok.
    """
    if not FACE_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Model face landmarker tidak ditemukan di: {FACE_LANDMARKER_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf
    )

    detector = vision.FaceLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return None, "Wajah tidak terdeteksi pada citra. Pastikan wajah tegak lurus tampak depan."

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image_rgb.shape

    # 1. Ekstraksi Seluruh 81 Koordinat Landmark
    points_81 = {}
    points_table = []

    for dlib_idx, (mp_idx, desc) in DLIB_81_MAP.items():
        lm = landmarks[mp_idx]
        px_x = float(lm.x * w)
        px_y = float(lm.y * h)
        points_81[dlib_idx] = np.array([px_x, px_y], dtype=np.float64)
        points_table.append({
            "Index": dlib_idx,
            "Nama Landmark": desc,
            "Koordinat X": round(px_x, 1),
            "Koordinat Y": round(px_y, 1)
        })

    def dist(idx1, idx2):
        return float(np.linalg.norm(points_81[idx1] - points_81[idx2]))

    # 2. Perhitungan Jarak Anatomis Sesuai Jurnal & Kebutuhan Kacamata
    # D1: Lebar Tulang Pipi / Lebar Wajah (Jarak antara titik 2 dan 14)
    d1 = dist(2, 14)
    # D2: Lebar Pelipis / Dahi (Jarak antara titik 75 dan 79)
    d2 = dist(75, 79)
    # D3: Panjang Wajah Vertikal (Midpoint titik 69 & 72 ke titik 8 dagu)
    p_hairline = (points_81[69] + points_81[72]) / 2.0
    d3 = float(np.linalg.norm(p_hairline - points_81[8]))
    # D4: Panjang Garis Rahang (Titik 8 ke 12)
    d4 = dist(8, 12)
    # D5: Lebar Rahang (Titik 4 ke 12)
    d5 = dist(4, 12)
    # D6: Lebar Rahang Bawah / Area Mulut (Titik 6 ke 10)
    d6 = dist(6, 10)
    # D7: Lebar Dasar Dagu (Titik 7 ke 9)
    d7 = dist(7, 9)

    # Jarak Mata & Hidung untuk Kacamata
    # Lebar Jembatan Hidung (Jarak sudut dalam mata kanan 39 ke kiri 42)
    d_bridge = dist(39, 42)
    # Interpupillary Distance (IPD): Pusat mata kanan ke pusat mata kiri
    p_eye_r = np.mean([points_81[i] for i in range(36, 42)], axis=0)
    p_eye_l = np.mean([points_81[i] for i in range(42, 48)], axis=0)
    d_ipd = float(np.linalg.norm(p_eye_r - p_eye_l))

    # Lebar Maksimum Wajah (Titik 0 ke 16)
    d_bizygomatic = dist(0, 16)

    measurements = {
        "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D7": d7,
        "Bridge_Width": d_bridge, "IPD": d_ipd, "Max_Face_Width": d_bizygomatic
    }

    # 3. Normalisasi Jarak (N1 - N7)
    d_sum = (d1 + d2 + d3 + d4 + d5 + d6 + d7) if (d1 + d2 + d3 + d4 + d5 + d6 + d7) > 0 else 1.0
    norm_distances = {f"N{i}": measurements[f"D{i}"] / d_sum for i in range(1, 8)}

    # 4. Perhitungan 10 Rasio Geometri (R1 - R10)
    r1 = d2 / (d1 if d1 > 0 else 1e-5)    # Lebar Dahi / Lebar Wajah
    r2 = d1 / (d3 if d3 > 0 else 1e-5)    # Lebar Wajah / Panjang Wajah
    r3 = d2 / (d3 if d3 > 0 else 1e-5)    # Lebar Dahi / Panjang Wajah
    r4 = d1 / (d5 if d5 > 0 else 1e-5)    # Lebar Wajah / Lebar Rahang
    r5 = d6 / (d5 if d5 > 0 else 1e-5)    # Lebar Bawah / Lebar Rahang
    r6 = d4 / (d6 if d6 > 0 else 1e-5)    # Garis Rahang / Lebar Bawah
    r7 = d6 / (d1 if d1 > 0 else 1e-5)    # Lebar Bawah / Lebar Wajah
    r8 = d5 / (d2 if d2 > 0 else 1e-5)    # Lebar Rahang / Lebar Dahi
    r9 = d4 / (d5 if d5 > 0 else 1e-5)    # Garis Rahang / Lebar Rahang
    r10 = d7 / (d6 if d6 > 0 else 1e-5)   # Lebar Dagu / Lebar Bawah

    ratios = {
        "R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5,
        "R6": r6, "R7": r7, "R8": r8, "R9": r9, "R10": r10
    }

    # 5. Perhitungan 3 Sudut Rahang (A1 - A3)
    v_hc = points_81[8] - p_hairline
    v_c10 = points_81[10] - points_81[8]
    v_c12 = points_81[12] - points_81[8]
    v_2_14 = points_81[14] - points_81[2]
    v_14_12 = points_81[12] - points_81[14]

    a1 = calculate_angle(-v_hc, v_c10)
    a2 = calculate_angle(-v_hc, v_c12)
    a3 = calculate_angle(v_2_14, v_14_12)

    angles = {"A1": a1, "A2": a2, "A3": a3}

    # 6. Ekstraksi 14 Fitur PCA Jurnal
    pca_features = {
        "R2": r2, "N5": norm_distances["N5"], "R3": r3, "A1": a1,
        "N2": norm_distances["N2"], "N3": norm_distances["N3"], "N6": norm_distances["N6"],
        "A2": a2, "R1": r1, "R8": r8, "R7": r7, "N1": norm_distances["N1"],
        "R10": r10, "R4": r4
    }

    # 7. Klasifikasi Bentuk Wajah (Heart, Oblong, Oval, Round, Square)
    pred_shape, probabilities = classify_shape(ratios, norm_distances, angles)

    # 8. Kalibrasi Skala Milimeter Berdasarkan Jarak Pupil (IPD Standar Antropometri ~63 mm)
    # IPD rata-rata manusia dewasa adalah 63 mm (rentang 58 - 70 mm)
    scale_mm_per_px = 63.0 / (d_ipd if d_ipd > 0 else 1.0)
    
    mm_temple_width = d2 * scale_mm_per_px       # Estimasi lebar pelipis dalam mm
    mm_cheek_width = d1 * scale_mm_per_px        # Estimasi lebar tulang pipi dalam mm
    mm_face_length = d3 * scale_mm_per_px        # Estimasi panjang wajah dalam mm
    mm_bridge_width = d_bridge * scale_mm_per_px  # Estimasi lebar jembatan hidung dalam mm
    mm_jaw_width = d5 * scale_mm_per_px          # Estimasi lebar rahang dalam mm

    mm_measurements = {
        "Temple_Width_mm": round(mm_temple_width, 1),
        "Cheekbone_Width_mm": round(mm_cheek_width, 1),
        "Face_Length_mm": round(mm_face_length, 1),
        "Bridge_Width_mm": round(mm_bridge_width, 1),
        "Jaw_Width_mm": round(mm_jaw_width, 1)
    }

    # 9. Rekomendasi Ukuran dan Model Kacamata Presisi
    glasses_recommendation = determine_glasses_size_and_style(
        mm_temple_width=mm_temple_width,
        mm_cheek_width=mm_cheek_width,
        mm_bridge_width=mm_bridge_width,
        r2_ratio=r2,
        face_shape=pred_shape
    )

    # 10. Pembuatan Citra Anotasi Lengkap 81 Titik (Persis Referensi Gambar)
    annotated_img_81 = create_81_landmark_visualization(image_rgb, points_81)

    result_data = {
        "predicted_shape": pred_shape,
        "probabilities": probabilities,
        "measurements_px": measurements,
        "measurements_mm": mm_measurements,
        "norm_distances": norm_distances,
        "ratios": ratios,
        "angles": angles,
        "pca_features": pca_features,
        "glasses": glasses_recommendation,
        "points_table": points_table,
        "points_81": points_81
    }

    return annotated_img_81, result_data


def classify_shape(r, n, a):
    """Mengklasifikasikan bentuk wajah berdasarkan aturan fitur PCA jurnal."""
    r1, r2, r3, r4, r8, r10 = r["R1"], r["R2"], r["R3"], r["R4"], r["R8"], r["R10"]
    a1, a2 = a["A1"], a["A2"]

    scores = {"Square": 0.0, "Round": 0.0, "Oval": 0.0, "Oblong": 0.0, "Heart": 0.0}

    # Square: R2 tinggi, R1 tinggi, R8 tinggi, R10 lebar
    scores["Square"] = (
        max(0.0, 1.0 - abs(r2 - 0.88) * 4.0) * 0.3 +
        max(0.0, 1.0 - abs(r1 - 0.90) * 4.0) * 0.25 +
        max(0.0, 1.0 - abs(r8 - 0.92) * 4.0) * 0.25 +
        min(1.0, max(0.0, (r10 - 1.2) * 1.5)) * 0.2
    )

    # Round: R2 tinggi, R1 lebih kecil dari 1, rahang melengkung
    scores["Round"] = (
        max(0.0, 1.0 - abs(r2 - 0.86) * 4.0) * 0.35 +
        max(0.0, 1.0 - abs(r1 - 0.76) * 4.0) * 0.25 +
        max(0.0, 1.0 - abs(r8 - 0.80) * 4.0) * 0.2 +
        max(0.0, 1.0 - abs(r10 - 1.1) * 3.0) * 0.2
    )

    # Oblong: R2 rendah (wajah panjang), R3 rendah
    ob_r2 = max(0.0, (0.75 - r2) * 5.0) if r2 < 0.75 else 0.0
    ob_r3 = max(0.0, (0.60 - r3) * 5.0) if r3 < 0.60 else 0.0
    scores["Oblong"] = min(1.0, (ob_r2 * 0.5) + (ob_r3 * 0.3) + max(0.0, 1.0 - abs(r8 - 0.85) * 3.0) * 0.2)

    # Heart: R1 tinggi (dahi lebar), R8 rendah (rahang lancip), sudut lancip
    ht_r1 = max(0.0, (r1 - 0.78) * 3.5)
    ht_r8 = max(0.0, (0.80 - r8) * 4.0) if r8 < 0.80 else 0.0
    scores["Heart"] = min(1.0, (ht_r1 * 0.35) + (ht_r8 * 0.4) + max(0.0, 1.0 - abs(a1 - 28.0) * 0.08) * 0.25)

    # Oval: Seimbang (Golden ratio)
    scores["Oval"] = (
        max(0.0, 1.0 - abs(r2 - 0.78) * 6.0) * 0.4 +
        max(0.0, 1.0 - abs(r1 - 0.82) * 5.0) * 0.3 +
        max(0.0, 1.0 - abs(r8 - 0.84) * 4.0) * 0.3
    )

    tot = sum(scores.values())
    if tot > 0:
        norm_scores = {k: round((v / tot) * 100, 1) for k, v in scores.items()}
    else:
        norm_scores = {k: 20.0 for k in scores}

    return max(norm_scores, key=norm_scores.get), norm_scores


def determine_glasses_size_and_style(mm_temple_width, mm_cheek_width, mm_bridge_width, r2_ratio, face_shape):
    """
    Menghitung rekomendasi ukuran kacamata standar optik (Lebar Frame Total, Lebar Lensa,
    Lebar Jembatan Hidung, Panjang Gagang, dan Tinggi Lensa) serta model bingkai terbaik.
    """
    # Lebar bingkai kacamata ideal harus setara atau sedikit melebihi lebar pelipis/tulang pipi (+2 s/d +4 mm)
    target_frame_width = max(mm_temple_width, mm_cheek_width) + 3.0

    # 1. Kategori Ukuran Kacamata (S / M / L / XL)
    if target_frame_width < 130.0:
        size_category = "S (Narrow / Wajah Kecil)"
        lens_width_range = (48, 50)
        temple_arm_length = 135
    elif 130.0 <= target_frame_width < 138.0:
        size_category = "M (Medium / Standar Proporsional)"
        lens_width_range = (51, 54)
        temple_arm_length = 140
    elif 138.0 <= target_frame_width < 145.0:
        size_category = "L (Wide / Wajah Lebar)"
        lens_width_range = (55, 58)
        temple_arm_length = 145
    else:
        size_category = "XL (Extra Wide / Sangat Lebar)"
        lens_width_range = (59, 62)
        temple_arm_length = 150

    # 2. Lebar Jembatan Hidung (Bridge Width)
    if mm_bridge_width < 17.0:
        bridge_category = "Narrow Bridge (15 - 16 mm)"
        rec_bridge_val = 16
    elif 17.0 <= mm_bridge_width < 20.0:
        bridge_category = "Medium Bridge (17 - 19 mm)"
        rec_bridge_val = 18
    else:
        bridge_category = "Wide Bridge (20 - 22 mm)"
        rec_bridge_val = 20

    # 3. Tinggi Lensa (Lens Height) Berdasarkan Rasio Proporsi Vertikal
    if r2_ratio < 0.74:
        # Wajah panjang (Oblong) -> Butuh lensa tinggi (Tall/Oversized)
        lens_height_desc = "Tinggi Lensa Besar / Tall Frame (>= 40 mm) untuk menyeimbangkan dimensi wajah panjang."
        rec_lens_height = 42
    elif r2_ratio > 0.85:
        # Wajah membulat/kotak -> Butuh lensa lebih ramping horizontal
        lens_height_desc = "Tinggi Lensa Sedang / Slim Frame (32 - 36 mm) untuk mempertegas garis horizontal."
        rec_lens_height = 34
    else:
        lens_height_desc = "Tinggi Lensa Standar (36 - 40 mm) dengan proporsi seimbang."
        rec_lens_height = 38

    # 4. Model Frame Berdasarkan Bentuk Wajah (Jurnal & Eyeglass Paper)
    styles = {
        "Square": {
            "best_frames": "Frame Bulat (Round), Oval, Cat-Eye Melengkung, atau Rimless Tipis.",
            "avoid_frames": "Frame Persegi Kaku dan Tebal.",
            "reason": "Sudut melengkung pada frame bulat melembutkan ketegasan garis rahang kotak."
        },
        "Round": {
            "best_frames": "Frame Persegi Panjang (Rectangular), Geometris Bersudut, Wayfarer Tajam.",
            "avoid_frames": "Frame Bulat Kecil atau Melingkar.",
            "reason": "Sudut tegas pada kacamata persegi memberikan struktur kontras pada pipi yang bulat."
        },
        "Oval": {
            "best_frames": "Hampir Semua Jenis Frame: Walnut, Aviator, Persegi Panjang, Retro Square.",
            "avoid_frames": "Frame yang lebarnya melebihi batas terluar tulang pipi.",
            "reason": "Proporsi wajah oval sangat seimbang sehingga cocok dengan beragam siluet bingkai."
        },
        "Oblong": {
            "best_frames": "Frame Oversized, Lensa Tinggi (Deep Frame), Bingkai Tebal, Wayfarer Besar.",
            "avoid_frames": "Frame Sempit/Pipih Horizontal Tanpa Tinggi Lensa.",
            "reason": "Frame tinggi dan tebal memotong siluet wajah panjang sehingga terlihat lebih proporsional."
        },
        "Heart": {
            "best_frames": "Frame Aviator, Frame Bottom-Heavy (aksen di bawah), Oval Lembut, Rimless.",
            "avoid_frames": "Frame dengan dekorasi berat di bagian atas alis (Heavy Browline).",
            "reason": "Menyeimbangkan dahi yang lebar dengan dagu yang lancip tanpa membebani bagian atas wajah."
        }
    }

    selected_style = styles.get(face_shape, styles["Oval"])

    # 5. Format Kode Optik Internasional: [Lebar Lensa] [] [Lebar Bridge] - [Panjang Tangkai]
    mean_lens = int((lens_width_range[0] + lens_width_range[1]) / 2)
    optical_code = f"{mean_lens} [] {rec_bridge_val} - {temple_arm_length} mm"

    return {
        "size_category": size_category,
        "recommended_frame_width": round(target_frame_width, 1),
        "optical_code": optical_code,
        "lens_width": f"{lens_width_range[0]} - {lens_width_range[1]} mm",
        "bridge_width": bridge_category,
        "temple_arm_length": f"{temple_arm_length} mm",
        "lens_height": lens_height_desc,
        "best_frames": selected_style["best_frames"],
        "avoid_frames": selected_style["avoid_frames"],
        "reason": selected_style["reason"]
    }


def create_geometric_visualization(image_rgb: np.ndarray, pts_81: dict) -> np.ndarray:
    """Menggambar garis jarak geometris D1..D7."""
    annotated = image_rgb.copy()
    def p(idx):
        return (int(pts_81[idx][0]), int(pts_81[idx][1]))
    cv2.line(annotated, p(2), p(14), (37, 99, 235), 2)
    cv2.line(annotated, p(75), p(79), (16, 185, 129), 2)
    p_hair = ((pts_81[69] + pts_81[72]) / 2.0).astype(int)
    cv2.line(annotated, (p_hair[0], p_hair[1]), p(8), (220, 38, 38), 2)
    cv2.line(annotated, p(8), p(12), (245, 158, 11), 2)
    cv2.line(annotated, p(4), p(12), (147, 51, 234), 2)
    cv2.line(annotated, p(6), p(10), (6, 182, 212), 2)
    cv2.line(annotated, p(7), p(9), (236, 72, 153), 2)
    for idx in [2, 14, 75, 79, 69, 72, 8, 4, 12, 6, 10, 7, 9]:
        cv2.circle(annotated, p(idx), 4, (0, 255, 255), -1)
    return annotated

def create_combined_visualization(image_rgb: np.ndarray, pts_81: dict) -> np.ndarray:
    """Menggabungkan garis jarak geometris D1..D7 dengan anotasi 81 titik landmark kuning & angka merah."""
    annotated = create_geometric_visualization(image_rgb, pts_81)
    return create_81_landmark_visualization(annotated, pts_81)

def create_81_landmark_visualization(image_rgb: np.ndarray, pts_81: dict) -> np.ndarray:
    """
    Menggambar seluruh 81 titik landmark dengan lingkaran kuning dan label angka merah
    persis seperti pada gambar referensi (uploaded_media) dan repository codeniko.
    """
    annotated = image_rgb.copy()
    h, w, _ = annotated.shape

    # Ukuran dinamis sesuai resolusi citra
    circle_radius = max(2, int(min(h, w) / 280))
    font_scale = max(0.28, min(h, w) / 2200)
    font_thick = 1

    # Warna
    COLOR_YELLOW = (0, 255, 255) # BGR: Kuning
    COLOR_RED = (0, 0, 255)      # BGR: Merah

    # 1. Gambar Garis Penghubung Kontur Tipis Transparan untuk Membantu Orientasi
    # Rahang: 0-16
    for i in range(16):
        pt1 = (int(pts_81[i][0]), int(pts_81[i][1]))
        pt2 = (int(pts_81[i+1][0]), int(pts_81[i+1][1]))
        cv2.line(annotated, pt1, pt2, (40, 40, 40), 1)

    # 2. Gambar 81 Titik Kuning & Teks Angka Merah
    for idx in range(81):
        if idx not in pts_81:
            continue
        cx = int(pts_81[idx][0])
        cy = int(pts_81[idx][1])

        # Titik lingkaran kuning
        cv2.circle(annotated, (cx, cy), circle_radius + 1, (0, 0, 0), -1)      # Outline hitam tipis
        cv2.circle(annotated, (cx, cy), circle_radius, COLOR_YELLOW, -1)     # Inti kuning

        # Teks indeks merah
        text_str = str(idx)
        # Posisi teks sedikit di atas atau kanan titik agar tidak menutupi lingkaran
        tx = cx + circle_radius + 2
        ty = cy - 1
        cv2.putText(
            annotated,
            text_str,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLOR_RED,
            font_thick,
            cv2.LINE_AA
        )

    return annotated
