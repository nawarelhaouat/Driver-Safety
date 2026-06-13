# =============================================================================
# evaluate_model.py
# -----------------------------------------------------------------------------
# Évalue le modèle entraîné sur les données de test.
# Affiche : matrice de confusion, rapport de classification, exemples visuels.
#
# Usage :
#   python evaluate_model.py
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

MODEL_PATH = "models/eye_cnn.h5"
DATA_DIR   = "data"
IMG_SIZE   = (64, 64)
CLASSES    = ["awake", "drowsy"]


def load_all_images():
    X, y, paths = [], [], []
    for label_id, label_name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, label_name)
        for fname in os.listdir(folder):
            if not fname.endswith(".jpg"):
                continue
            path = os.path.join(folder, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img  = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_id)
            paths.append(path)
    X = np.array(X, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]
    return X, np.array(y), paths


def evaluate():
    print("Chargement du modèle...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Chargement des données...")
    X, y_true, paths = load_all_images()

    print("Prédictions...")
    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # --- Rapport de classification ---
    print("\n--- Rapport de classification ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Matrice de confusion")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()
    print("Matrice sauvegardée : models/confusion_matrix.png")

    # --- Exemples d'erreurs ---
    errors = [(i, y_true[i], y_pred[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    if errors:
        print(f"\n{len(errors)} erreurs de classification.")
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for ax, (i, true, pred) in zip(axes.flat, errors[:10]):
            img = (X[i, :, :, 0] * 255).astype(np.uint8)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Réel:{CLASSES[true]}\nPrédit:{CLASSES[pred]}", fontsize=8, color='red')
            ax.axis('off')
        plt.suptitle("Exemples d'erreurs")
        plt.tight_layout()
        plt.savefig("models/errors.png")
        plt.show()


if __name__ == "__main__":
    evaluate()