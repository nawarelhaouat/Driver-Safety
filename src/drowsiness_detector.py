# =============================================================================
# src/drowsiness_detector.py — version PyTorch
# =============================================================================

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from config.settings import CONSECUTIVE_FRAMES, EAR_THRESHOLD

MODEL_PATH = os.path.join("models", "eye_cnn.pth")
IMG_SIZE   = (64, 64)


class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),          nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DrowsinessDetector:

    def __init__(self, consecutive_frames=CONSECUTIVE_FRAMES):
        self.consecutive_frames = consecutive_frames
        self.frame_counter      = 0
        self.is_drowsy          = False
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(MODEL_PATH):
            self.model = EyeCNN().to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            self._use_cnn = True
            print("[DrowsinessDetector] Modèle CNN PyTorch chargé.")
        else:
            self.model    = None
            self._use_cnn = False
            print("[DrowsinessDetector] Modèle absent → fallback seuil EAR.")
            print("                     Lancez : python train_model.py")

    def _predict_eye(self, eye_points, frame):
        xs = [p[0] for p in eye_points]
        ys = [p[1] for p in eye_points]
        x1 = max(0, min(xs) - 10)
        x2 = min(frame.shape[1], max(xs) + 10)
        y1 = max(0, min(ys) - 10)
        y2 = min(frame.shape[0], max(ys) + 10)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return 0

        eye_crop    = frame[y1:y2, x1:x2]
        eye_gray    = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        eye_resized = cv2.resize(eye_gray, IMG_SIZE).astype(np.float32) / 255.0

        # (1, 1, 64, 64) — batch=1, canal=1
        tensor = torch.tensor(eye_resized).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            pred   = output.argmax(1).item()  # 0=awake, 1=drowsy

        return pred

    def update(self, ear_value, frame=None, left_eye_pts=None, right_eye_pts=None):
        if self._use_cnn and frame is not None and left_eye_pts and right_eye_pts:
            pred_left  = self._predict_eye(left_eye_pts,  frame)
            pred_right = self._predict_eye(right_eye_pts, frame)
            is_closed  = (pred_left == 1 and pred_right == 1)
        else:
            is_closed = ear_value < EAR_THRESHOLD

        if is_closed:
            self.frame_counter += 1
            if self.frame_counter >= self.consecutive_frames:
                self.is_drowsy = True
        else:
            self.frame_counter = 0
            self.is_drowsy     = False

        return self.is_drowsy

    def get_status_text(self):
        return "DROWSY !" if self.is_drowsy else "NORMAL"

    def get_frame_counter(self):
        return self.frame_counter

    def reset(self):
        self.frame_counter = 0
        self.is_drowsy     = False