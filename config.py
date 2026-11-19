import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
EXTERNAL_MODELS_DIR = PARENT_DIR / "mediapipe-model-mirrors-1"
JURNAL_DIR = PARENT_DIR / "jurnal"
TUGAS_AKHIR_DIR = BASE_DIR / "Tugas Akhir" / "Tugas Akhir"

# Model Paths
FACE_LANDMARKER_PATH = EXTERNAL_MODELS_DIR / "face_landmarker.task"
HAND_LANDMARKER_PATH = EXTERNAL_MODELS_DIR / "hand_landmarker.task"
PDF_JURNAL_PATH = JURNAL_DIR / "Data_Mining_Project3.pdf"

# Tugas Akhir Trained ML Models & Dataset
DATASET_CSV_PATH = TUGAS_AKHIR_DIR / "output" / "face_shape_features.csv"
TRAINED_ML_MODEL_PATH = MODELS_DIR / "face_shape_ensemble_model.joblib"
TRAINED_LR_MODEL_PATH = MODELS_DIR / "face_shape_model.joblib"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# App Metadata
APP_TITLE = "Face Shape & Vision Studio"
APP_SUBTITLE = "Klasifikasi Bentuk Wajah Berbasis Landmark Geometris & Rekomendasi Kacamata"
APP_ICON = ""
VERSION = "2.1.0"

# Journal & Training Metadata
JOURNAL_TITLE = "Human Face Shape Classification with Machine Learning"
JOURNAL_AUTHORS = "Ashwinee Mehta, Taha Mohamed Mahmoud"
JOURNAL_CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]
DATASET_SAMPLES_COUNT = 596

# Default Processing Configurations
IMAGE_MAX_WIDTH = 800
DEFAULT_CANNY_THRESHOLD_1 = 100
DEFAULT_CANNY_THRESHOLD_2 = 200
