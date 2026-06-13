# =============================================================================
# config/settings.py
# -----------------------------------------------------------------------------
# Ce fichier centralise toutes les constantes du projet.
# Modifier ces valeurs pour ajuster le comportement du système.
# =============================================================================

# --- Paramètres de la caméra ---
CAMERA_INDEX = 0          # 0 = webcam par défaut ; changer à 1, 2... si plusieurs caméras

# --- Seuil EAR (Eye Aspect Ratio) ---
# L'EAR est une valeur entre 0 et 1.
# Un œil grand ouvert → EAR ≈ 0.3 ou plus
# Un œil à moitié fermé → EAR ≈ 0.2
# Un œil fermé         → EAR ≈ 0.1 ou moins
EAR_THRESHOLD = 0.15      # En dessous de ce seuil → œil considéré comme fermé

# --- Nombre de frames consécutives avant alerte ---
# Si l'EAR reste sous le seuil pendant N frames, on déclenche l'alerte.
# À 30 fps, 10 frames ≈ 0.33 secondes de somnolence
CONSECUTIVE_FRAMES = 10

# --- Chemin vers le fichier audio de l'alarme ---
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALARM_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")

# --- Paramètres d'affichage ---
FRAME_WIDTH  = 800        # Largeur de la fenêtre OpenCV
FRAME_HEIGHT = 600        # Hauteur de la fenêtre OpenCV

# --- Couleurs BGR pour OpenCV ---
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
