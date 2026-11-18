import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
MODELS_DIR = PARENT_DIR / "mediapipe-model-mirrors-1"
JURNAL_DIR = PARENT_DIR / "jurnal"

# Model Paths
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
PDF_JURNAL_PATH = JURNAL_DIR / "Data_Mining_Project3.pdf"

# App Metadata
APP_TITLE = "VisionLab Studio"
APP_SUBTITLE = "Implementasi Klasifikasi Bentuk Wajah Berbasis Landmark Geometris & PCD"
APP_ICON = None
VERSION = "1.1.0"

# Journal Reference Metadata
JOURNAL_TITLE = "Human Face Shape Classification with Machine Learning"
JOURNAL_AUTHORS = "Ashwinee Mehta, Taha Mohamed Mahmoud"
JOURNAL_CLASSES = ["Heart", "Oblong", "Oval", "Round", "Square"]

# Default Processing Configurations
IMAGE_MAX_WIDTH = 800
DEFAULT_CANNY_THRESHOLD_1 = 100
DEFAULT_CANNY_THRESHOLD_2 = 200
