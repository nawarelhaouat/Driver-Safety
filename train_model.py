# =============================================================================
# train_model.py  — version PyTorch (compatible Python 3.14)
# =============================================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Paramètres ---
IMG_SIZE   = (64, 64)
BATCH_SIZE = 32
EPOCHS     = 30
DATA_DIR   = "data"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "eye_cnn.pth")
CLASSES    = ["awake", "drowsy"]

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilisé : {device}")


# =============================================================================
# 1. Dataset PyTorch
# =============================================================================
class EyeDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images  = images
        self.labels  = labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
        ]) if augment else transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = (self.images[idx] * 255).astype(np.uint8)
        img   = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# =============================================================================
# 2. Architecture CNN
# =============================================================================
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Bloc 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Bloc 3
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


# =============================================================================
# 3. Chargement des données
# =============================================================================
def load_dataset():
    X, y = [], []
    for label_id, label_name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            print(f"[ERREUR] Dossier manquant : {folder}")
            print(f"         Lancez : python collect_data.py --label {label_name}")
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        print(f"  Classe '{label_name}' : {len(files)} images")
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_id)

    X = np.array(X, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]  # (N, 64, 64, 1)
    y = np.array(y, dtype=np.int64)
    return shuffle(X, y, random_state=42)


# =============================================================================
# 4. Entraînement
# =============================================================================
def train():
    print("\n" + "="*55)
    print("  Entraînement CNN (PyTorch) — Eye Drowsiness Classifier")
    print("="*55)

    print("\n[1/4] Chargement des données...")
    X, y = load_dataset()
    if len(X) == 0:
        print("[ERREUR] Aucune donnée trouvée. Collectez d'abord les images.")
        return

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[2/4] Création des DataLoaders...")
    train_loader = DataLoader(EyeDataset(X_train, y_train, augment=True),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(EyeDataset(X_val,   y_val,   augment=False), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(EyeDataset(X_test,  y_test,  augment=False), batch_size=BATCH_SIZE)

    print("\n[3/4] Construction du modèle...")
    model     = EyeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    print("\n[4/4] Entraînement...")
    best_val_acc = 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(labels)
        train_acc  = correct / total
        train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs   = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += len(labels)
        val_acc  = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        scheduler.step(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train acc: {train_acc:.3f} loss: {train_loss:.4f} | "
              f"Val acc: {val_acc:.3f} loss: {val_loss:.4f}")

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"             → Modèle sauvegardé (val_acc={val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("  Early stopping déclenché.")
                break

    # --- Évaluation finale ---
    print("\n--- Évaluation sur le jeu de test ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Matrice de confusion")
    plt.ylabel("Réel") ; plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    print("  Matrice sauvegardée : models/confusion_matrix.png")

    # Courbes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_acc"], label="Train") ; ax1.plot(history["val_acc"], label="Val")
    ax1.set_title("Accuracy") ; ax1.legend()
    ax2.plot(history["train_loss"], label="Train") ; ax2.plot(history["val_loss"], label="Val")
    ax2.set_title("Loss") ; ax2.legend()
    plt.tight_layout()
    plt.savefig("models/training_curves.png")
    print("  Courbes sauvegardées : models/training_curves.png")
    plt.show()


if __name__ == "__main__":
    train()