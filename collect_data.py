# =============================================================================
# collect_data.py
# -----------------------------------------------------------------------------
# Script de collecte de données d'entraînement.
# Il ouvre la webcam, détecte les yeux, et sauvegarde les images rognées
# des yeux dans data/awake/ ou data/drowsy/.
#
# Usage :
#   python collect_data.py --label awake  --samples 500
#   python collect_data.py --label drowsy --samples 500
# =============================================================================

import cv2
import os
import argparse
import sys
from src.face_detector import FaceDetector
from src.eye_detector  import get_average_ear
from config.settings   import CAMERA_INDEX

# --- Arguments ligne de commande ---
parser = argparse.ArgumentParser(description="Collecte de données pour le CNN")
parser.add_argument("--label",   required=True, choices=["awake", "drowsy"],
                    help="Classe à collecter : awake ou drowsy")
parser.add_argument("--samples", type=int, default=500,
                    help="Nombre d'images à collecter (défaut : 500)")
args = parser.parse_args()

LABEL      = args.label
N_SAMPLES  = args.samples
SAVE_DIR   = os.path.join("data", LABEL)
IMG_SIZE   = (64, 64)   # taille uniforme des images sauvegardées

os.makedirs(SAVE_DIR, exist_ok=True)

# Compter les images déjà existantes pour ne pas écraser
existing = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
count    = existing

print("=" * 55)
print(f"  Collecte : classe '{LABEL}'")
print(f"  Objectif : {N_SAMPLES} images  |  Déjà collectées : {existing}")
print(f"  Dossier  : {SAVE_DIR}/")
print("=" * 55)
print("  ESPACE  → capturer une image")
print("  A       → mode automatique (capture continue)")
print("  Q       → quitter")
print()

detector   = FaceDetector()
cap        = cv2.VideoCapture(CAMERA_INDEX)
auto_mode  = False   # mode capture automatique


def extract_eye_region(frame, eye_points, margin=10):
    """
    Extrait et redimensionne la région autour de l'œil.
    Retourne une image en niveaux de gris de taille IMG_SIZE.
    """
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]

    x1 = max(0, min(xs) - margin)
    x2 = min(frame.shape[1], max(xs) + margin)
    y1 = max(0, min(ys) - margin)
    y2 = min(frame.shape[0], max(ys) + margin)

    # Sécurité : région trop petite → ignorer
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    eye_crop = frame[y1:y2, x1:x2]
    eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    eye_resized = cv2.resize(eye_gray, IMG_SIZE)
    return eye_resized


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    display   = frame.copy()
    landmarks = detector.detect(frame)

    eye_img_left  = None
    eye_img_right = None

    if landmarks:
        coords = detector.get_pixel_coordinates(landmarks, frame.shape)
        ear, left_pts, right_pts = get_average_ear(coords)

        eye_img_left  = extract_eye_region(frame, left_pts)
        eye_img_right = extract_eye_region(frame, right_pts)

        # Dessiner les points des yeux
        for (x, y) in left_pts + right_pts:
            cv2.circle(display, (x, y), 2, (0, 255, 0), -1)

        # Afficher EAR
        cv2.putText(display, f"EAR: {ear:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Barre de statut
    color = (0, 200, 0) if LABEL == "awake" else (0, 0, 220)
    cv2.rectangle(display, (0, 0), (display.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(display, f"Classe: {LABEL.upper()}  |  {count}/{N_SAMPLES}  |  {'[AUTO]' if auto_mode else '[MANUEL]'}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Aperçu de l'œil extrait
    if eye_img_left is not None:
        preview = cv2.resize(eye_img_left, (80, 80))
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        display[50:130, display.shape[1]-90:display.shape[1]-10] = preview_bgr

    cv2.imshow("Collecte de données", display)

    key = cv2.waitKey(30 if auto_mode else 1) & 0xFF

    # --- Sauvegarde ---
    def save_eyes():
        global count
        saved = False
        for eye_img in [eye_img_left, eye_img_right]:
            if eye_img is not None and count < N_SAMPLES:
                path = os.path.join(SAVE_DIR, f"{count:05d}.jpg")
                cv2.imwrite(path, eye_img)
                count += 1
                saved = True
        if saved:
            print(f"  [{count}/{N_SAMPLES}] Images sauvegardées", end="\r")

    if key == ord(' ') and landmarks:
        save_eyes()

    elif key == ord('a'):
        auto_mode = not auto_mode
        print(f"\n  Mode automatique : {'ON' if auto_mode else 'OFF'}")

    elif key == ord('q'):
        break

    # En mode auto, capturer à chaque frame
    if auto_mode and landmarks:
        save_eyes()

    # Objectif atteint
    if count >= N_SAMPLES:
        print(f"\n  Objectif atteint : {count} images collectées !")
        break

cap.release()
cv2.destroyAllWindows()
detector.release()
print(f"\n  Collecte terminée. Images dans : {SAVE_DIR}/")