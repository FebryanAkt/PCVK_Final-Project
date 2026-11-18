import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import (
    FACE_LANDMARKER_PATH,
    TRAINED_ML_MODEL_PATH,
    TRAINED_LR_MODEL_PATH,
    SELECTED_FEATURES_PATH,
    LABEL_ENCODER_PATH,
    DATASET_CSV_PATH
)

# ==============================================================================
# PEMETAAN 81 TITIK LANDMARK (CODENIKO SHAPE PREDICTOR 81) KE MEDIAPIPE FACE MESH
# ==============================================================================

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

def angle_at(a, b, c):
    """Sudut (derajat) pada titik b, antara vektor b->a dan b->c (Persis formula notebook Tugas Akhir)."""
    v1, v2 = a - b, c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosv = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

# Cache model terlatih agar cepat saat inferensi berulang
_CACHED_ML_MODEL = None
_CACHED_FEATURES = None
_CACHED_LE = None

def load_trained_ml_model():
    """Memuat model machine learning terlatih dari folder models / Tugas Akhir."""
    global _CACHED_ML_MODEL, _CACHED_FEATURES, _CACHED_LE
    if _CACHED_ML_MODEL is not None:
        return _CACHED_ML_MODEL, _CACHED_FEATURES, _CACHED_LE

    model_target = TRAINED_ML_MODEL_PATH if TRAINED_ML_MODEL_PATH.exists() else TRAINED_LR_MODEL_PATH

    if model_target.exists() and SELECTED_FEATURES_PATH.exists() and LABEL_ENCODER_PATH.exists():
        try:
            _CACHED_ML_MODEL = joblib.load(model_target)
            _CACHED_FEATURES = joblib.load(SELECTED_FEATURES_PATH)
            _CACHED_LE = joblib.load(LABEL_ENCODER_PATH)
            return _CACHED_ML_MODEL, _CACHED_FEATURES, _CACHED_LE
        except Exception as e:
            print("Gagal memuat model tersimpan:", e)
    return None, None, None


def analyze_face_shape_and_glasses(image_rgb: np.ndarray, min_conf: float = 0.5):
    """
    Ekstraksi 81 landmark, perhitungan jarak anatomis (pelipis, pipi, rahang, jembatan hidung),
    dan inferensi bentuk wajah menggunakan MODEL MACHINE LEARNING HASIL TRAINING DATASET TUGAS AKHIR.
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

    # 2. Perhitungan Jarak (D1 - D7)
    d1 = dist(2, 14)                                        # Lebar tulang pipi (D1)
    d2 = dist(75, 79)                                       # Lebar pelipis / dahi (D2)
    p_hair = (points_81[69] + points_81[72]) / 2.0          # Garis rambut
    p_chin = points_81[8]                                   # Ujung dagu
    d3 = float(np.linalg.norm(p_hair - p_chin))             # Panjang wajah (D3)
    d4 = dist(8, 12)                                        # Panjang garis rahang (D4)
    d5 = dist(4, 12)                                        # Lebar sudut rahang (D5)
    d6 = dist(6, 10)                                        # Lebar rahang bawah (D6)
    d7 = dist(7, 9)                                         # Lebar dasar dagu (D7)

    # Jarak Mata & Hidung untuk Kacamata
    d_bridge = dist(39, 42)
    p_eye_r = np.mean([points_81[i] for i in range(36, 42)], axis=0)
    p_eye_l = np.mean([points_81[i] for i in range(42, 48)], axis=0)
    d_ipd = float(np.linalg.norm(p_eye_r - p_eye_l))
    d_bizygomatic = dist(0, 16)

    measurements = {
        "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D7": d7,
        "Bridge_Width": d_bridge, "IPD": d_ipd, "Max_Face_Width": d_bizygomatic
    }

    # 3. Normalisasi Jarak (D1_norm - D7_norm) Persis Formula Tugas Akhir
    d_raw = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D7": d7}
    total_d = sum(d_raw.values()) + 1e-9
    norm_distances = {f"{k}_norm": v / total_d for k, v in d_raw.items()}

    # 4. Perhitungan 10 Rasio Geometri (R1 - R10)
    ratios = {
        "R1": d2 / (d1 + 1e-9),
        "R2": d1 / (d3 + 1e-9),
        "R3": d2 / (d3 + 1e-9),
        "R4": d1 / (d5 + 1e-9),
        "R5": d6 / (d5 + 1e-9),
        "R6": d4 / (d6 + 1e-9),
        "R7": d6 / (d1 + 1e-9),
        "R8": d5 / (d2 + 1e-9),
        "R9": d4 / (d5 + 1e-9),
        "R10": d7 / (d6 + 1e-9)
    }

    # 5. Perhitungan 3 Sudut (A1 - A3) Persis Formula Tugas Akhir
    a1 = angle_at(p_hair, p_chin, points_81[10])
    a2 = angle_at(p_hair, p_chin, points_81[12])
    a3 = angle_at(points_81[2], points_81[14], points_81[12])
    angles = {"A1": a1, "A2": a2, "A3": a3}

    # Gabungkan seluruh 20 fitur untuk input model ML
    all_features_dict = {}
    all_features_dict.update(norm_distances)
    all_features_dict.update(ratios)
    all_features_dict.update(angles)

    # 6. INFERENSI DENGAN MODEL MACHINE LEARNING HASIL TRAINING DATASET TUGAS AKHIR
    ml_model, selected_features, le = load_trained_ml_model()

    if ml_model is not None and selected_features is not None and le is not None:
        # Gunakan model terlatih (Ensemble / Logistic Regression) dari dataset 596 data
        x_input = pd.DataFrame([all_features_dict])[selected_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        probs_raw = ml_model.predict_proba(x_input)[0]
        
        # Mapping probabilitas ke kelas
        probabilities = {}
        for cls_name, prob_val in zip(le.classes_, probs_raw):
            formatted_cls = cls_name.capitalize()
            probabilities[formatted_cls] = round(float(prob_val) * 100, 1)

        pred_shape = max(probabilities, key=probabilities.get)
        model_source = "Machine Learning Model (Ensemble Trained on Tugas Akhir Dataset)"
    else:
        # Fallback aturan geometris jika file model belum ter-load
        pred_shape, probabilities = fallback_rule_classifier(ratios, norm_distances, angles)
        model_source = "Geometric Rules Classifier (Baseline)"

    # 7. Kalibrasi Skala Milimeter Berdasarkan Antropometri Optik (IPD ~ 63 mm)
    scale_mm_per_px = 63.0 / (d_ipd if d_ipd > 0 else 1.0)
    mm_temple_width = d2 * scale_mm_per_px
    mm_cheek_width = d1 * scale_mm_per_px
    mm_face_length = d3 * scale_mm_per_px
    mm_bridge_width = d_bridge * scale_mm_per_px
    mm_jaw_width = d5 * scale_mm_per_px

    mm_measurements = {
        "Temple_Width_mm": round(mm_temple_width, 1),
        "Cheekbone_Width_mm": round(mm_cheek_width, 1),
        "Face_Length_mm": round(mm_face_length, 1),
        "Bridge_Width_mm": round(mm_bridge_width, 1),
        "Jaw_Width_mm": round(mm_jaw_width, 1)
    }

    # 8. Rekomendasi Ukuran dan Model Kacamata
    glasses_recommendation = determine_glasses_size_and_style(
        mm_temple_width=mm_temple_width,
        mm_cheek_width=mm_cheek_width,
        mm_bridge_width=mm_bridge_width,
        r2_ratio=ratios["R2"],
        face_shape=pred_shape
    )

    # 9. Anotasi Citra 81 Titik
    annotated_img_81 = create_81_landmark_visualization(image_rgb, points_81)

    result_data = {
        "predicted_shape": pred_shape,
        "probabilities": probabilities,
        "model_source": model_source,
        "measurements_px": measurements,
        "measurements_mm": mm_measurements,
        "norm_distances": norm_distances,
        "ratios": ratios,
        "angles": angles,
        "pca_features": {k: all_features_dict[k] for k in (selected_features if selected_features else ratios.keys())},
        "glasses": glasses_recommendation,
        "points_table": points_table,
        "points_81": points_81
    }

    return annotated_img_81, result_data


def fallback_rule_classifier(r, n, a):
    """Fallback classifier berbasis aturan geometris."""
    r1, r2, r3, r4, r8, r10 = r["R1"], r["R2"], r["R3"], r["R4"], r["R8"], r["R10"]
    scores = {"Square": 0.0, "Round": 0.0, "Oval": 0.0, "Oblong": 0.0, "Heart": 0.0}
    scores["Square"] = (max(0.0, 1.0 - abs(r2 - 0.88) * 4.0) * 0.3 + max(0.0, 1.0 - abs(r1 - 0.90) * 4.0) * 0.25 + min(1.0, max(0.0, (r10 - 1.2) * 1.5)) * 0.2)
    scores["Round"] = (max(0.0, 1.0 - abs(r2 - 0.86) * 4.0) * 0.35 + max(0.0, 1.0 - abs(r1 - 0.76) * 4.0) * 0.25 + max(0.0, 1.0 - abs(r10 - 1.1) * 3.0) * 0.2)
    scores["Oblong"] = min(1.0, (max(0.0, (0.75 - r2) * 5.0) if r2 < 0.75 else 0.0) * 0.5 + max(0.0, 1.0 - abs(r8 - 0.85) * 3.0) * 0.2)
    scores["Heart"] = min(1.0, (max(0.0, (r1 - 0.78) * 3.5)) * 0.35 + (max(0.0, (0.80 - r8) * 4.0) if r8 < 0.80 else 0.0) * 0.4)
    scores["Oval"] = (max(0.0, 1.0 - abs(r2 - 0.78) * 6.0) * 0.4 + max(0.0, 1.0 - abs(r1 - 0.82) * 5.0) * 0.3 + max(0.0, 1.0 - abs(r8 - 0.84) * 4.0) * 0.3)
    tot = sum(scores.values())
    norm_scores = {k: round((v / tot) * 100, 1) if tot > 0 else 20.0 for k, v in scores.items()}
    return max(norm_scores, key=norm_scores.get), norm_scores


def determine_glasses_size_and_style(mm_temple_width, mm_cheek_width, mm_bridge_width, r2_ratio, face_shape):
    """Menghitung ukuran kacamata presisi standar optik dan rekomendasi bentuk bingkai."""
    target_frame_width = max(mm_temple_width, mm_cheek_width) + 3.0

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

    if mm_bridge_width < 17.0:
        bridge_category = "Narrow Bridge (15 - 16 mm)"
        rec_bridge_val = 16
    elif 17.0 <= mm_bridge_width < 20.0:
        bridge_category = "Medium Bridge (17 - 19 mm)"
        rec_bridge_val = 18
    else:
        bridge_category = "Wide Bridge (20 - 22 mm)"
        rec_bridge_val = 20

    if r2_ratio < 0.74:
        lens_height_desc = "Tinggi Lensa Besar / Tall Frame (>= 40 mm) untuk memotong kesan wajah panjang."
    elif r2_ratio > 0.85:
        lens_height_desc = "Tinggi Lensa Sedang / Slim Frame (32 - 36 mm) agar proporsi wajah lebih terstruktur."
    else:
        lens_height_desc = "Tinggi Lensa Standar (36 - 40 mm) dengan siluet proporsional."

    styles = {
        "Square": {
            "best_frames": "Frame Bulat (Round), Oval, Cat-Eye Melengkung, atau Rimless Tipis.",
            "avoid_frames": "Frame Persegi Kaku dan Tebal.",
            "reason": "Sudut melengkung pada frame bulat melembutkan ketegasan garis rahang kotak."
        },
        "Round": {
            "best_frames": "Frame Persegi Panjang (Rectangular), Geometris Bersudut, Wayfarer Tajam.",
            "avoid_frames": "Frame Bulat Kecil atau Melingkar.",
            "reason": "Sudut tegas pada kacamata persegi memberikan struktur kontras pada pipi yang membulat."
        },
        "Oval": {
            "best_frames": "Hampir Semua Jenis Frame: Walnut, Aviator, Persegi Panjang, Retro Square.",
            "avoid_frames": "Frame yang lebarnya melebihi batas terluar tulang pipi.",
            "reason": "Proporsi wajah oval sangat seimbang sehingga cocok dengan beragam bentuk bingkai."
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
    """Menggambar seluruh 81 titik landmark kuning dan angka merah (0-80)."""
    annotated = image_rgb.copy()
    h, w, _ = annotated.shape

    circle_radius = max(2, int(min(h, w) / 280))
    font_scale = max(0.28, min(h, w) / 2200)
    font_thick = 1

    COLOR_YELLOW = (0, 255, 255)
    COLOR_RED = (0, 0, 255)

    for i in range(16):
        pt1 = (int(pts_81[i][0]), int(pts_81[i][1]))
        pt2 = (int(pts_81[i+1][0]), int(pts_81[i+1][1]))
        cv2.line(annotated, pt1, pt2, (40, 40, 40), 1)

    for idx in range(81):
        if idx not in pts_81:
            continue
        cx = int(pts_81[idx][0])
        cy = int(pts_81[idx][1])
        cv2.circle(annotated, (cx, cy), circle_radius + 1, (0, 0, 0), -1)
        cv2.circle(annotated, (cx, cy), circle_radius, COLOR_YELLOW, -1)
        cv2.putText(
            annotated,
            str(idx),
            (cx + circle_radius + 2, cy - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLOR_RED,
            font_thick,
            cv2.LINE_AA
        )

    return annotated
