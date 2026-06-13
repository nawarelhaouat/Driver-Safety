# =============================================================================
# src/eye_detector.py
# -----------------------------------------------------------------------------
# Responsabilité : extraire les points des yeux depuis les landmarks du visage
# et calculer l'Eye Aspect Ratio (EAR).
#
# La formule EAR mesure l'ouverture de l'œil :
#
#          ||p2 - p6|| + ||p3 - p5||
#  EAR = ──────────────────────────────
#                2 × ||p1 - p4||
#
# Où p1..p6 sont les 6 points du contour de l'œil :
#   p1 = coin gauche, p4 = coin droit
#   p2, p3 = points hauts, p5, p6 = points bas
#
# Référence des indices MediaPipe Face Mesh (468 landmarks) :
#   Œil gauche  : [33, 160, 158, 133, 153, 144]
#   Œil droit   : [362, 385, 387, 263, 373, 380]
# =============================================================================

from src.utils import euclidean_distance


# --- Indices MediaPipe pour les yeux ---
# Ces indices correspondent aux 6 points caractéristiques de chaque œil
# dans la grille de 468 landmarks de MediaPipe Face Mesh.

LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def get_eye_points(landmarks_coords, eye_indices):
    """
    Extrait les coordonnées pixel des points d'un œil depuis la liste complète
    des landmarks.

    Paramètres :
        landmarks_coords : liste de tuples (x, y) — tous les landmarks du visage
        eye_indices      : liste d'entiers — indices des points de l'œil

    Retourne :
        list of tuples : coordonnées (x, y) des points de l'œil
    """
    return [landmarks_coords[i] for i in eye_indices]


def calculate_ear(eye_points):
    """
    Calcule l'Eye Aspect Ratio (EAR) à partir des 6 points de l'œil.

    Disposition des points (dans l'ordre des indices) :
        [0] = p1 (coin gauche / externe)
        [1] = p2 (haut gauche)
        [2] = p3 (haut droit)
        [3] = p4 (coin droit / interne)
        [4] = p5 (bas droit)
        [5] = p6 (bas gauche)

    Paramètres :
        eye_points : liste de 6 tuples (x, y)

    Retourne :
        float : valeur EAR (entre 0 et ~0.4 en pratique)
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    # Distance verticale 1 : entre le haut gauche (p2) et le bas gauche (p6)
    vertical_1 = euclidean_distance(p2, p6)

    # Distance verticale 2 : entre le haut droit (p3) et le bas droit (p5)
    vertical_2 = euclidean_distance(p3, p5)

    # Distance horizontale : entre le coin gauche (p1) et le coin droit (p4)
    horizontal = euclidean_distance(p1, p4)

    # Formule EAR — évite la division par zéro avec un epsilon
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)

    return ear


def get_average_ear(landmarks_coords):
    """
    Calcule l'EAR moyen des deux yeux pour plus de robustesse.

    En pratique, on fait la moyenne de l'EAR gauche et droit,
    car un seul œil peut être partiellement masqué.

    Paramètres :
        landmarks_coords : liste de tuples (x, y) pour tous les landmarks

    Retourne :
        tuple : (ear_moyenne, points_oeil_gauche, points_oeil_droit)
    """
    # Extraction des points de chaque œil
    left_eye_pts  = get_eye_points(landmarks_coords, LEFT_EYE_INDICES)
    right_eye_pts = get_eye_points(landmarks_coords, RIGHT_EYE_INDICES)

    # Calcul EAR pour chaque œil
    left_ear  = calculate_ear(left_eye_pts)
    right_ear = calculate_ear(right_eye_pts)

    # Moyenne des deux EAR
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear, left_eye_pts, right_eye_pts
