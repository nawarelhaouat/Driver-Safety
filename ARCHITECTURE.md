# ARCHITECTURE - Drowsy Driver Safety Alert System

##  Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Architecture générale](#architecture-générale)
3. [Modules détaillés](#modules-détaillés)
4. [Flux de traitement](#flux-de-traitement)
5. [Configuration](#configuration)
6. [Diagramme de flux](#diagramme-de-flux)

---

## Vue d'ensemble

Le **Drowsy Driver Safety Alert System** est un système temps réel qui détecte la somnolence d'un conducteur via sa webcam et déclenche une alarme automatique.

### Objectifs principaux :
- Capturer le flux vidéo en continu
- Détecter le visage et les yeux
- Mesurer l'ouverture des yeux (Eye Aspect Ratio)
- Identifier les signes de somnolence
- Déclencher une alarme sonore et visuelle

---

## Architecture générale

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN.PY (Orchestrateur)                   │
│            Boucle principale temps réel (30fps)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
            ┌──────────────────┼──────────────────┐
            ↓                  ↓                  ↓
    ┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐
    │ FACE_DETECTOR   │ │ EYE_DETECTOR  │ │ DROWSINESS_      │
    │ (MediaPipe)     │ │ (EAR Calcul)  │ │ DETECTOR (CNN/   │
    │                 │ │               │ │ Threshold)       │
    │ • Détecte 468   │ │ • Extrait 6   │ │                  │
    │   landmarks du  │ │   points/œil  │ │ • Compte frames  │
    │   visage        │ │ • Calcule EAR │ │ • Décide si      │
    │ • Retourne      │ │   moyenne     │ │   DROWSY/AWAKE   │
    │   coords pixels │ │               │ │                  │
    └─────────────────┘ └───────────────┘ └──────────────────┘
            ↓                  ↓                  ↓
            └──────────────────┼──────────────────┘
                              ↓
                    ┌──────────────────┐
                    │ ALARM_MANAGER    │
                    │ (Pygame)         │
                    │                  │
                    │ • Play alarm.wav │
                    │ • Stop alarm     │
                    └──────────────────┘
                              ↓
                    ┌──────────────────┐
                    │ UTILS.PY         │
                    │ (Affichage)      │
                    │                  │
                    │ • Dessiner       │
                    │   landmarks      │
                    │ • Barre statut   │
                    │ • Textes alertes │
                    └──────────────────┘
                              ↓
                    ┌──────────────────┐
                    │ OpenCV (Display) │
                    │ → Écran utilisat │
                    └──────────────────┘
```

---

## Modules détaillés

###  **main.py** - Orchestrateur principal

**Responsabilité :** Coordonne tous les composants et gère la boucle vidéo temps réel.

**Processus :**
```python
BOUCLE INFINIE (30 fps) :
  1. Capture frame webcam
  2. Miroir horizontal (comme un miroir)
  3. Détecte le visage
     ├─ Si visage détecté :
     │   ├─ Extrait landmarks (468 points)
     │   ├─ Calcule EAR (Eye Aspect Ratio)
     │   ├─ Décide DROWSY/AWAKE
     │   ├─ Affiche landmarks en VERT (awake) ou ROUGE (drowsy)
     │   ├─ Joue alarme si DROWSY
     │   └─ Affiche "!!! DROWSINESS ALERT !!!"
     └─ Si aucun visage :
        ├─ Arrête l'alarme
        ├─ Réinitialise compteurs
        └─ Affiche "Aucun visage détecté"
  4. Affiche barre statut (EAR, STATUS, FRAME COUNT)
  5. Gère touches clavier :
     └─ 'q' : Quitter
     └─ 'r' : Réinitialiser
```

**Fichier :** [`main.py`](main.py)

---

###  **config/settings.py** - Configuration centralisée

**Responsabilité :** Centralise TOUS les paramètres ajustables.

**Paramètres clés :**

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `CAMERA_INDEX` | 0 | Index caméra (0 = webcam par défaut) |
| `EAR_THRESHOLD` | 0.15 | Seuil EAR pour détecter yeux fermés |
| `CONSECUTIVE_FRAMES` | 10 | Frames consécutives = ~0.33 sec à 30fps |
| `FRAME_WIDTH` | 800 | Largeur fenêtre OpenCV |
| `FRAME_HEIGHT` | 600 | Hauteur fenêtre OpenCV |
| `ALARM_PATH` | assets/alarm.wav | Chemin fichier audio |

**Avantage :** Modifier un paramètre ici affecte tout le système sans toucher au code principal !

**Fichier :** [`config/settings.py`](config/settings.py)

---

###  **src/face_detector.py** - Détection du visage

**Technologie :** MediaPipe Face Mesh (468 landmarks)

**Ce qu'il fait :**
```python
ENTRÉE : Frame (image 3D BGR)
  ↓
Utilise MediaPipe pour détecter les 468 points du visage
  ↓
SORTIE : Landmarks (coordonnées normalisées 0-1)
```

**Points détectés :**
- Contour du visage (points 0-16)
- Yeux gauche/droit (points 33, 133, 362, 263 etc.)
- Sourcils, lèvres, nez, etc.

**Méthodes principales :**
```python
face_detector = FaceDetector()

# Détecte et retourne landmarks normalisés
landmarks = face_detector.detect(frame)

# Convertit coordonnées normalisées → pixels
coords = face_detector.get_pixel_coordinates(landmarks, frame.shape)
```

**Fichier :** [`src/face_detector.py`](src/face_detector.py)

---

### **src/eye_detector.py** - Extraction et calcul EAR

**Technologie :** Formule Eye Aspect Ratio (Soukupová & Tereza, 2016)

**Concept EAR :**
```
        ||p2 - p6|| + ||p3 - p5||
EAR = ───────────────────────────
           2 × ||p1 - p4||

Où p1..p6 = 6 points de contour de l'œil
```

**Interprétation :**
| EAR | État | Exemple |
|-----|------|---------|
| > 0.25 | AWAKE (éveillé) | Yeux bien ouverts |
| 0.15 - 0.25 | Intermédiaire | Yeux à moitié fermés |
| < 0.15 | DROWSY (somnolent) | Yeux fermés ❌ |

**Indices MediaPipe pour les yeux :**
```
Œil gauche  : indices [33, 160, 158, 133, 153, 144]
Œil droit   : indices [362, 385, 387, 263, 373, 380]
```

**Processus :**
```python
1. Extrait 6 points de l'œil gauche
2. Extrait 6 points de l'œil droit
3. Calcule EAR gauche et droit
4. Retourne moyenne(EAR_gauche, EAR_droit)
```

**Fichier :** [`src/eye_detector.py`](src/eye_detector.py)

---

###  **src/drowsiness_detector.py** - Décision somnolence

**Modes de fonctionnement :**

#### Mode 1 : Fallback EAR (Actuel ✅)
```python
if EAR < THRESHOLD (0.15) :
    frame_counter += 1
    if frame_counter >= 10 :
        is_drowsy = TRUE → ALARME !
else :
    frame_counter = 0
    is_drowsy = FALSE
```

**Exemple :**
```
Frame 1-9 : EAR < 0.15 → Counter = 1,2,3...9
Frame 10 : EAR < 0.15 → Counter = 10 → ALARM! 🚨
Frame 11 : EAR > 0.15 → Counter = 0 → STOP ALARM
```

#### Mode 2 : CNN (Optional, non disponible en Python 3.14)
- Utilise un réseau de neurones entraîné
- Plus précis mais nécessite TensorFlow

**Fichier :** [`src/drowsiness_detector.py`](src/drowsiness_detector.py)

---

###  **src/alarm_manager.py** - Gestion alarme

**Responsabilité :** Jouer/arrêter le son d'alarme.

**Utilise :** Pygame mixer pour audio

**Méthodes :**
```python
alarm = AlarmManager(alarm_path="assets/alarm.wav")

alarm.play()    # Démarre l'alarme (ne relance pas si déjà active)
alarm.stop()    # Arrête l'alarme
alarm.release() # Libère ressources
```

**Fichier :** [`src/alarm_manager.py`](src/alarm_manager.py)

---

###  **src/utils.py** - Utilitaires affichage

**Fonctions :**
```python
draw_eye_landmarks(frame, points, color)
    └─ Dessine les 6 points de l'œil avec cercles

draw_text(frame, text, position, font_scale, color, thickness)
    └─ Affiche du texte sur la frame

draw_status_bar(frame, ear_value, status, frame_count, threshold)
    └─ Barre statut en haut :
       "Classe: AWAKE | EAR: 0.279 | 0/10 | Threshold: 0.15"

euclidean_distance(p1, p2)
    └─ Calcule distance entre 2 points
```

**Fichier :** [`src/utils.py`](src/utils.py)

---

###  **generate_alarm.py** - Génération audio

**Responsabilité :** Créer le fichier `alarm.wav` par synthèse sonore.

**Processus :**
```
Génère une tonalité beep-beep-beep
Sauvegarde dans assets/alarm.wav
```

**Utilisation :**
```bash
python generate_alarm.py
```

**Fichier :** [`generate_alarm.py`](generate_alarm.py)

---

###  **collect_data.py** - Collecte données training

**Objectif :** Collecter des images d'yeux fermés/ouverts pour entraîner un CNN.

**Usage :**
```bash
# Collecter 500 images yeux OUVERTS
python collect_data.py --label awake --samples 500

# Collecter 500 images yeux FERMÉS
python collect_data.py --label drowsy --samples 500
```

**Contrôles :**
| Touche | Action |
|--------|--------|
| ESPACE | Capture manuelle (2 images : gauche + droit) |
| A | Mode automatique (capture chaque frame) |
| Q | Quitter |

**Output :** Images sauvegardées dans `data/awake/` et `data/drowsy/`

**Fichier :** [`collect_data.py`](collect_data.py)

---

###  **train_model.py** - Entraînement CNN

**Objectif :** Entraîner un réseau de neurones convolutionnel sur les images collectées.

**Architecture :**
```
INPUT (64x64x1 grayscale)
  ↓
Conv2D(32) + MaxPool + Batch Norm
  ↓
Conv2D(64) + MaxPool + Batch Norm
  ↓
Conv2D(128) + MaxPool + Batch Norm
  ↓
Flatten + Dense(256) + Dropout(0.5)
  ↓
OUTPUT Dense(2) → [p_awake, p_drowsy]
```

**Prerequis :** TensorFlow 2.13+ (Non disponible en Python 3.14)

**Fichier :** [`train_model.py`](train_model.py)

---

##  Flux de traitement

### Flux complet d'une frame :

```
1. CAPTURE
   └─ cap.read() → frame (résolution caméra)

2. PRÉTRAITEMENT
   └─ cv2.flip(frame, 1) → Miroir horizontal

3. DÉTECTION VISAGE
   └─ face_detector.detect(frame)
      └─ MediaPipe retourne 468 landmarks normalisés
      └─ Convertir en coordonnées pixels

4. CALCUL EAR
   └─ get_average_ear(coords)
      ├─ Extrait 6 points œil gauche
      ├─ Extrait 6 points œil droit
      ├─ Calcule distance verticale/horizontale
      ├─ Retourne EAR_moyen

5. DÉCISION SOMNOLENCE
   └─ drowsiness_detector.update(ear)
      ├─ Si EAR < 0.15 :
      │  └─ Augmente frame_counter
      │     └─ Si frame_counter >= 10 :
      │        └─ is_drowsy = TRUE
      └─ Sinon :
         └─ Réinitialise frame_counter

6. ALARME
   └─ Si is_drowsy :
      ├─ alarm.play() → Joue alarm.wav
      └─ Affiche texte rouge "!!! DROWSINESS ALERT !!!"
   └─ Sinon :
      └─ alarm.stop()

7. AFFICHAGE
   └─ Dessine landmarks (VERT ou ROUGE)
   └─ Affiche barre statut
   └─ Affiche frame à l'écran

8. GESTION CLAVIER
   └─ 'q' : Quitter
   └─ 'r' : Réinitialiser
```

---

## Configuration

### Fichier : `config/settings.py`

**Pour rendre la détection plus sensible :**
```python
EAR_THRESHOLD = 0.12      # Plus bas = plus sensible
CONSECUTIVE_FRAMES = 5    # Plus bas = alerte plus rapide (0.17 sec)
```

**Pour rendre moins sensible :**
```python
EAR_THRESHOLD = 0.20      # Plus haut = moins sensible
CONSECUTIVE_FRAMES = 20   # Plus haut = alerte plus lente (0.67 sec)
```

**Pour changer caméra :**
```python
CAMERA_INDEX = 1          # Utilise 2ème caméra au lieu de 0
```

**Pour changer résolution affichage :**
```python
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
```

---

##  Diagramme de flux (Texte)

```
                    WEBCAM
                      ↓
              ┌───────────────┐
              │  OPENCV: cap  │
              │  .read() frame│
              └───────────────┘
                      ↓
              ┌───────────────┐
              │  PRÉTRAITEMENT│
              │  cv2.flip(1)  │
              └───────────────┘
                      ↓
    ┌─────────────────────────────────────────┐
    │        FACE_DETECTOR (MediaPipe)         │
    │  Détecte 468 landmarks du visage        │
    └─────────────────────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │   Landmarks found?     │
         └────────────────────────┘
           /                        \
          OUI                        NON
          ↓                          ↓
    ┌──────────────┐          ┌────────────────┐
    │ EYE_DETECTOR │          │  alarm.stop()  │
    │ Calcule EAR  │          │  reset()       │
    └──────────────┘          │ display: "pas  │
           ↓                   │ de visage"     │
    ┌──────────────┐          └────────────────┘
    │ DROWSINESS   │                │
    │ DETECTOR     │                │
    │ Compare EAR  │                │
    │ vs THRESHOLD │                │
    └──────────────┘                │
         ↓                           │
    ┌─────────────┐                 │
    │ is_drowsy?  │                 │
    └─────────────┘                 │
     /             \                │
   OUI             NON              │
    ↓               ↓               │
┌────────┐     ┌─────────┐         │
│alarm.  │     │alarm.   │         │
│play()  │     │stop()   │         │
│display │     │display  │         │
│ALERT!  │     │"AWAKE"  │         │
└────────┘     └─────────┘         │
    ↓               ↓               │
    └───────────────┴───────────────┘
                    ↓
         ┌──────────────────────┐
         │  UTILS: AFFICHAGE    │
         │ • landmarks (vert/   │
         │   rouge)             │
         │ • Barre statut       │
         │ • Textes alertes     │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  OPENCV: imshow()    │
         │  Affiche frame final │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │  GESTION CLAVIER     │
         │  • 'q' : Quitter     │
         │  • 'r' : Réinitial.  │
         └──────────────────────┘
                    ↓
            Boucle suivante
```

---

## Résumé structure

```
DrowsyDriverSafetyAlertSystem/
│
├── main.py                       ← POINT D'ENTRÉE (orchestrateur)
├── generate_alarm.py             ← Génère alarm.wav
├── collect_data.py               ← Collecte données training
├── train_model.py                ← Entraîne CNN (TensorFlow)
├── requirements.txt              ← Dépendances
│
├── config/                       ← Configuration
│   ├── __init__.py
│   └── settings.py               ← Paramètres centralisés 
│
├── src/                          ← Cœur du système
│   ├── __init__.py
│   ├── face_detector.py          ← Détecte visage (MediaPipe)
│   ├── eye_detector.py           ← Calcule EAR
│   ├── drowsiness_detector.py    ← Décide DROWSY/AWAKE
│   ├── alarm_manager.py          ← Joue alarme (Pygame)
│   └── utils.py                  ← Fonctions affichage
│
├── assets/                       ← Ressources
│   └── alarm.wav                 ← Son d'alarme
│
└── data/                         ← Données training (créé par collect_data.py)
    ├── awake/                    ← Images yeux ouverts
    └── drowsy/                   ← Images yeux fermés
```

---

##  Démarrage rapide

### 1️⃣ Installation
```bash
pip install -r requirements.txt
```

### 2️⃣ Générer alarme (si alarm.wav n'existe pas)
```bash
python generate_alarm.py
```

### 3️⃣ Lancer le système
```bash
python main.py
```

### 4️⃣ Tester (Fermez les yeux ~0.33 sec)
```
Yeux ouverts → EAR ≈ 0.3 → "AWAKE" → Pas d'alarme 
Yeux fermés  → EAR ≈ 0.1 → Compteur monte → "DROWSY" → ALARME! 
Yeux rouverts → Compteur reset → Alarme stop 
```

---

## Performances

| Métrique | Valeur |
|----------|--------|
| FPS (frames/sec) | 30 |
| Latence détection | ~33ms |
| Temps avant alarme | ~0.33 sec (10 frames) |
| Caméra requise | Webcam standard USB/intégrée |
| CPU | Faible (~5-10%) |
| RAM | ~150MB |

---

**Date:** 2026  
**Status:**  Fonctionnel (Fallback EAR)
