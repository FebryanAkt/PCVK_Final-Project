import os
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import FACE_LANDMARKER_PATH, HAND_LANDMARKER_PATH

# Landmark connections for Hand (21 points)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

class VisionModelManager:
    """Manager untuk memuat dan menjalankan model MediaPipe Tasks."""
    
    @staticmethod
    def is_face_model_available() -> bool:
        return FACE_LANDMARKER_PATH.exists()

    @staticmethod
    def is_hand_model_available() -> bool:
        return HAND_LANDMARKER_PATH.exists()

    @staticmethod
    def detect_face_landmarks(image_rgb: np.ndarray, num_faces: int = 1, min_conf: float = 0.5):
        """
        Deteksi landmark wajah menggunakan MediaPipe FaceLandmarker.
        Mengembalikan tuple: (annotated_image, result_dict)
        """
        if not VisionModelManager.is_face_model_available():
            raise FileNotFoundError(f"Model face landmarker tidak ditemukan di: {FACE_LANDMARKER_PATH}")

        base_options = python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf
        )

        detector = vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        annotated = image_rgb.copy()
        h, w, _ = annotated.shape
        face_count = len(detection_result.face_landmarks)

        for landmarks in detection_result.face_landmarks:
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 0), -1)

        info = {
            "faces_detected": face_count,
            "has_landmarks": face_count > 0
        }
        return annotated, info

    @staticmethod
    def detect_hand_landmarks(image_rgb: np.ndarray, num_hands: int = 2, min_conf: float = 0.5):
        """
        Deteksi landmark tangan menggunakan MediaPipe HandLandmarker.
        Mengembalikan tuple: (annotated_image, result_dict)
        """
        if not VisionModelManager.is_hand_model_available():
            raise FileNotFoundError(f"Model hand landmarker tidak ditemukan di: {HAND_LANDMARKER_PATH}")

        base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf
        )

        detector = vision.HandLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        annotated = image_rgb.copy()
        h, w, _ = annotated.shape
        hand_count = len(detection_result.hand_landmarks)
        handedness_list = []

        if detection_result.handedness:
            for hand_category in detection_result.handedness:
                if len(hand_category) > 0:
                    handedness_list.append(hand_category[0].category_name)

        for hand_lms in detection_result.hand_landmarks:
            # Draw Connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = (int(hand_lms[start_idx].x * w), int(hand_lms[start_idx].y * h))
                pt2 = (int(hand_lms[end_idx].x * w), int(hand_lms[end_idx].y * h))
                cv2.line(annotated, pt1, pt2, (255, 200, 0), 2)
            
            # Draw Keypoints
            for idx, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (255, 0, 0) if idx in [4, 8, 12, 16, 20] else (0, 255, 128)
                cv2.circle(annotated, (cx, cy), 4, color, -1)

        info = {
            "hands_detected": hand_count,
            "handedness": handedness_list
        }
        return annotated, info

    @staticmethod
    def detect_face_opencv(image_rgb: np.ndarray):
        """Deteksi wajah fallback menggunakan OpenCV Haar Cascade bawaan."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        annotated = image_rgb.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Wajah", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info = {
            "faces_detected": len(faces)
        }
        return annotated, info
