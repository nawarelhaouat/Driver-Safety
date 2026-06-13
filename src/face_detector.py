# =============================================================================
# src/face_detector.py
# -----------------------------------------------------------------------------
# Détection du visage compatible avec mediapipe >= 0.10 (nouvelle API Tasks)
# ET l'ancienne API mp.solutions (< 0.10).
# La bonne API est détectée automatiquement au démarrage.
# =============================================================================

import cv2
import os
import urllib.request
import mediapipe as mp

# Détection automatique de l'API disponible
try:
    _ = mp.solutions.face_mesh
    _USE_TASKS_API = False
    print("[FaceDetector] Utilisation de l'ancienne API mediapipe (solutions).")
except AttributeError:
    _USE_TASKS_API = True
    print("[FaceDetector] Utilisation de la nouvelle API mediapipe (Tasks).")

# Chemin local du modèle (nouvelle API seulement)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _download_model_if_needed():
    if not os.path.exists(_MODEL_PATH):
        print("[FaceDetector] Téléchargement du modèle (~29 Mo), patientez...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[FaceDetector] Modèle téléchargé avec succès.")


class _LandmarkWrapper:
    """Adaptateur : expose .landmark[] comme l'ancienne API."""
    def __init__(self, landmark_list):
        self.landmark = landmark_list


class FaceDetector:

    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if not _USE_TASKS_API:
            self._mode = "solutions"
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self._mode = "tasks"
            _download_model_if_needed()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._frame_ts = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._mode == "solutions":
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0]
            return None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_ts += 33
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
            if result.face_landmarks:
                return _LandmarkWrapper(result.face_landmarks[0])
            return None

    def get_pixel_coordinates(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

    def release(self):
        if self._mode == "solutions":
            self.face_mesh.close()
        else:
            self._landmarker.close()