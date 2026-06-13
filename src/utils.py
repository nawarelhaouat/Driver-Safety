# =============================================================================
# src/utils.py
# -----------------------------------------------------------------------------
# Fonctions utilitaires partagées dans tout le projet :
#   - calcul de distance euclidienne entre deux points
#   - affichage de texte enrichi sur une image OpenCV
#   - dessin des landmarks (points du visage) sur l'image
# =============================================================================

import cv2
import numpy as np


def euclidean_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points 2D.

    Paramètres :
        point1 (tuple ou array) : premier point (x1, y1)
        point2 (tuple ou array) : deuxième point (x2, y2)

    Retourne :
        float : la distance entre les deux points
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def draw_text(frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """
    Affiche du texte sur une image OpenCV avec un fond semi-transparent
    pour améliorer la lisibilité.

    Paramètres :
        frame     : l'image (tableau NumPy)
        text      : le texte à afficher
        position  : tuple (x, y) en pixels
        font_scale : taille de la police
        color     : couleur BGR du texte
        thickness : épaisseur du texte
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calcul de la taille du texte pour dessiner le fond
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # Rectangle de fond légèrement transparent (aplati sur l'image)
    cv2.rectangle(
        frame,
        (x - 4, y - text_height - 4),
        (x + text_width + 4, y + baseline),
        (0, 0, 0),
        cv2.FILLED
    )

    # Texte par-dessus le fond
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_eye_landmarks(frame, eye_points, color=(0, 255, 0)):
    """
    Dessine les points de l'œil et les relie par un contour.

    Paramètres :
        frame      : l'image sur laquelle dessiner
        eye_points : liste de tuples (x, y) représentant les points de l'œil
        color      : couleur BGR du dessin
    """
    # Conversion en tableau NumPy pour polylines
    pts = np.array(eye_points, dtype=np.int32)

    # Dessine le contour fermé de l'œil
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

    # Dessine chaque point individuellement
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 2, color, -1)


def draw_status_bar(frame, ear_value, status, frame_count, threshold):
    """
    Affiche une barre d'informations en haut de l'écran avec :
      - la valeur EAR
      - le statut (Normal / Drowsy)
      - le nombre de frames somnolentes consécutives

    Paramètres :
        frame       : l'image OpenCV
        ear_value   : float, valeur courante de l'EAR
        status      : str, "NORMAL" ou "DROWSY"
        frame_count : int, nombre de frames consécutives sous le seuil
        threshold   : float, le seuil EAR configuré
    """
    h, w = frame.shape[:2]

    # Fond de la barre de statut
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), cv2.FILLED)

    # Couleur selon le statut
    status_color = (0, 255, 0) if status == "NORMAL" else (0, 0, 255)

    # Affichage des informations
    draw_text(frame, f"EAR : {ear_value:.3f}  (seuil : {threshold})",
              (10, 30), font_scale=0.65, color=(200, 200, 200))
    draw_text(frame, f"Status : {status}",
              (10, 60), font_scale=0.8, color=status_color, thickness=2)
    draw_text(frame, f"Frames somnolentes : {frame_count}",
              (w // 2, 30), font_scale=0.6, color=(180, 180, 180))
