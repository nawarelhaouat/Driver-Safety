# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# Point d'entrée principal du système de détection de somnolence.
#
# Ce script :
#   1. Ouvre la webcam
#   2. Détecte le visage avec MediaPipe (FaceDetector)
#   3. Extrait les points des yeux et calcule l'EAR (EyeDetector)
#   4. Décide si le conducteur est somnolent (DrowsinessDetector)
#   5. Déclenche/arrête l'alarme sonore (AlarmManager)
#   6. Affiche tout en temps réel avec OpenCV
#
# Contrôles :
#   - Appuyer sur 'q' pour quitter
#   - Appuyer sur 'r' pour réinitialiser le détecteur
# =============================================================================

import cv2
import sys
import os

# Ajout du dossier racine au path Python pour les imports relatifs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import des modules du projet
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    ALARM_PATH, EAR_THRESHOLD, COLOR_RED, COLOR_GREEN, COLOR_WHITE
)
from src.face_detector      import FaceDetector
from src.eye_detector       import get_average_ear
from src.drowsiness_detector import DrowsinessDetector
from src.alarm_manager      import AlarmManager
from src.utils              import draw_eye_landmarks, draw_text, draw_status_bar


def main():
    """
    Fonction principale : boucle vidéo temps réel de détection de somnolence.
    """

    print("=" * 55)
    print("   Drowsy Driver Safety Alert System")
    print("   Appuyez sur 'q' pour quitter")
    print("   Appuyez sur 'r' pour réinitialiser")
    print("=" * 55)

    # -------------------------------------------------------------------------
    # Initialisation des composants
    # -------------------------------------------------------------------------
    face_detector      = FaceDetector()
    drowsiness_detector = DrowsinessDetector()
    alarm_manager      = AlarmManager(ALARM_PATH)

    # -------------------------------------------------------------------------
    # Ouverture de la webcam
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la caméra (index {CAMERA_INDEX}).")
        print("         Vérifiez que votre webcam est connectée et disponible.")
        sys.exit(1)

    # Résolution souhaitée (peut être ignorée selon la caméra)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[INFO] Webcam ouverte avec succès. Démarrage de la détection...")

    # -------------------------------------------------------------------------
    # Boucle principale de traitement vidéo
    # -------------------------------------------------------------------------
    while True:

        # --- Lecture d'une frame ---
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Impossible de lire la frame. Vérifiez la caméra.")
            break

        # Miroir horizontal pour un affichage plus naturel (comme un miroir)
        frame = cv2.flip(frame, 1)

        # Valeurs par défaut si aucun visage n'est détecté
        ear_value = 0.0
        status    = "AUCUN VISAGE"
        is_drowsy = False

        # --- Détection du visage ---
        landmarks = face_detector.detect(frame)

        if landmarks is not None:
            # Conversion des landmarks en coordonnées pixel
            coords = face_detector.get_pixel_coordinates(landmarks, frame.shape)

            # --- Calcul de l'EAR ---
            ear_value, left_eye_pts, right_eye_pts = get_average_ear(coords)

            # --- Décision somnolence ---
            is_drowsy = drowsiness_detector.update(ear_value, frame, left_eye_pts, right_eye_pts)
            status    = drowsiness_detector.get_status_text()

            # --- Dessin des contours des yeux ---
            eye_color = COLOR_RED if is_drowsy else COLOR_GREEN
            draw_eye_landmarks(frame, left_eye_pts,  color=eye_color)
            draw_eye_landmarks(frame, right_eye_pts, color=eye_color)

            # --- Alerte visuelle si somnolent ---
            if is_drowsy:
                # Grand message rouge clignotant au centre de l'écran
                h, w = frame.shape[:2]
                draw_text(
                    frame,
                    "!!! DROWSINESS ALERT !!!",
                    position=(w // 2 - 200, h // 2),
                    font_scale=1.2,
                    color=COLOR_RED,
                    thickness=3
                )

                # Déclencher l'alarme sonore (ne se relance pas si déjà active)
                alarm_manager.play()
            else:
                # Arrêter l'alarme si le conducteur est réveillé
                alarm_manager.stop()

        else:
            # Aucun visage détecté → on stoppe l'alarme et on affiche un message
            alarm_manager.stop()
            drowsiness_detector.reset()
            draw_text(frame, "Aucun visage detecte", (10, 120),
                      font_scale=0.7, color=COLOR_WHITE)

        # --- Barre de statut en haut de l'écran ---
        draw_status_bar(
            frame,
            ear_value=ear_value,
            status=status,
            frame_count=drowsiness_detector.get_frame_counter(),
            threshold=EAR_THRESHOLD
        )

        # --- Affichage de la frame finale ---
        cv2.imshow("Drowsy Driver Safety Alert System", frame)

        # --- Gestion des touches clavier ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 'q' → quitter proprement
            print("[INFO] Fermeture du programme...")
            break

        elif key == ord('r'):
            # 'r' → réinitialiser le détecteur de somnolence
            drowsiness_detector.reset()
            alarm_manager.stop()
            print("[INFO] Détecteur réinitialisé.")

    # -------------------------------------------------------------------------
    # Nettoyage des ressources
    # -------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    face_detector.release()
    alarm_manager.release()

    print("[INFO] Programme terminé proprement.")


# Point d'entrée Python
if __name__ == "__main__":
    main()
