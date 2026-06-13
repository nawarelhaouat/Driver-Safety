# 🚗 Drowsy Driver Safety Alert System

> Système de détection de somnolence en temps réel via webcam — Projet Image Processing

---

## 📋 Présentation

Ce projet implémente un **système de détection de somnolence pour conducteurs** basé sur la vision par ordinateur. Grâce à la webcam, il analyse le visage du conducteur en temps réel, mesure l'ouverture de ses yeux, et déclenche une alarme sonore et visuelle dès qu'une somnolence est détectée.

---

## 🎯 Objectif

Détecter automatiquement les signes de somnolence d'un conducteur en :
1. Capturant le flux vidéo de la webcam
2. Détectant les points clés du visage (landmarks)
3. Calculant l'**Eye Aspect Ratio (EAR)** pour chaque frame
4. Déclenchant une alerte si l'EAR reste trop bas pendant plusieurs frames consécutives

---

## 🛠 Technologies utilisées

| Bibliothèque | Rôle |
|---|---|
| **OpenCV** | Lecture webcam, affichage, dessin |
| **MediaPipe** | Détection des 468 landmarks du visage |
| **NumPy** | Calculs de distances et tableaux |
| **Pygame** | Lecture du son d'alarme |

---

## 📁 Structure du projet

```
DrowsyDriverSafetyAlertSystem/
│
├── main.py                    # Point d'entrée principal
├── generate_alarm.py          # Script utilitaire : génère alarm.wav
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
│
├── assets/
│   └── alarm.wav              # Fichier audio d'alarme
│
├── src/
│   ├── __init__.py
│   ├── face_detector.py       # Détection du visage (MediaPipe)
│   ├── eye_detector.py        # Extraction des yeux + calcul EAR
│   ├── drowsiness_detector.py # Logique de décision somnolence
│   ├── alarm_manager.py       # Gestion de l'alarme sonore
│   └── utils.py               # Fonctions utilitaires (affichage, dessin)
│
└── config/
    ├── __init__.py
    └── settings.py            # Constantes configurables
```

---

## ⚙️ Installation

### Prérequis

- Python **3.8 ou supérieur**
- Une webcam fonctionnelle

### Étapes

```bash
# 1. Cloner ou télécharger le projet
cd DrowsyDriverSafetyAlertSystem

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Générer le fichier alarm.wav (si absent)
python generate_alarm.py
```

---

## ▶️ Lancement

```bash
python main.py
```

### Contrôles clavier

| Touche | Action |
|--------|--------|
| `q` | Quitter le programme |
| `r` | Réinitialiser le détecteur |

---

## 🧮 Méthode EAR — Eye Aspect Ratio

L'**EAR** est une mesure géométrique de l'ouverture de l'œil, définie par la formule :

```
         ||P2 - P6|| + ||P3 - P5||
EAR  =  ─────────────────────────────
                2 × ||P1 - P4||
```

Où `P1` à `P6` sont les 6 points caractéristiques du contour de l'œil :

```
        P2   P3
   P1 ·       · P4
        P6   P5
```

| Situation | EAR approximatif |
|-----------|-----------------|
| Œil grand ouvert | ≈ 0.30 – 0.40 |
| Œil à moitié fermé | ≈ 0.20 – 0.25 |
| Œil fermé | ≈ 0.10 ou moins |

**Décision :** Si l'EAR moyen (des deux yeux) reste **inférieur à 0.22** pendant **20 frames consécutives** (≈ 0.67 secondes à 30 fps), le système déclare l'état **DROWSY** et déclenche l'alarme.

Ces valeurs sont configurables dans `config/settings.py`.

---

## 🔊 Fichier audio

Si le fichier `assets/alarm.wav` est absent, générez-le automatiquement :

```bash
python generate_alarm.py
```

Vous pouvez aussi placer **votre propre fichier** `.wav` dans `assets/` et le renommer en `alarm.wav`.  
Téléchargement gratuit : [freesound.org](https://freesound.org) (rechercher "alarm beep").

---

## ⚠️ Limites du projet

- **Conditions d'éclairage** : les performances baissent en faible luminosité
- **Angle du visage** : une vue de face est nécessaire ; le profil n'est pas bien géré
- **Lunettes** : peuvent perturber la détection des yeux selon le type de monture
- **Faux positifs** : les clignements normaux rapides peuvent rarement déclencher une alerte
- **Pas de détection de bâillement** : la bouche n'est pas analysée dans cette version

---

## 🚀 Améliorations possibles

1. **Détection du bâillement** — analyser l'ouverture de la bouche avec les landmarks de la bouche
2. **Calibration automatique** — ajuster le seuil EAR selon le conducteur au démarrage
3. **Enregistrement des événements** — journaliser les alertes avec horodatage
4. **Interface graphique Tkinter** — ajouter un tableau de bord avec statistiques
5. **Détection de la tête qui penche** — détecter la posture de la tête (head pose estimation)
6. **Modèle deep learning** — remplacer l'EAR par un classifieur CNN pour plus de robustesse
7. **Mode nuit** — traitement d'image adapté aux conditions de faible luminosité

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre du module **Image Processing**.

---

## 📄 Licence

Usage académique et éducatif uniquement.
