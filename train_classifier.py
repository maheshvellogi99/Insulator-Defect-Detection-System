"""
Two-Stage Insulator Classifier (EfficientNet-B0)
==================================================
Stage 2 of the detection-classification pipeline:
  1. YOLOv8m detects insulator bounding boxes
  2. EfficientNet-B0 classifies cropped regions as Normal/Defective

Training this classifier on cropped insulator regions from the CPLID dataset.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import random
from collections import defaultdict

# --- Configuration ---
RANDOM_SEED = 42
DATA_PATH = './yolo_insulator_project/data_v2'
MODEL_SAVE_PATH = './runs/classifier'
NUM_EPOCHS = 30
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-4
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InsulatorCropDataset(Dataset):
    """Dataset that crops insulator regions from images using YOLO labels.

    Class 0 (normal_insulator) → label 0
    Class 1 (defective_insulator) → label 1
    """

    def __init__(self, split, transform=None):
        self.transform = transform
        self.samples = []  # (crop_image_path_or_array, label)

        img_dir = os.path.join(DATA_PATH, 'images', split)
        lbl_dir = os.path.join(DATA_PATH, 'labels', split)

        for lbl_file in sorted(os.listdir(lbl_dir)):
            if not lbl_file.endswith('.txt'):
                continue

            img_file = lbl_file.replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, img_file)

            if not os.path.exists(img_path):
                continue

            with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(float(parts[0]))

                    # Only interested in insulator-level boxes (class 0 and 1)
                    # Skip defect_region (class 2) — those are too small for classification
                    if cls_id not in [0, 1]:
                        continue

                    x_center, y_center, w, h = map(float, parts[1:])
                    # Binary label: 0=normal, 1=defective
                    label = 0 if cls_id == 0 else 1
                    self.samples.append((img_path, x_center, y_center, w, h, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x_center, y_center, w, h, label = self.samples[idx]

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Return a blank image if reading fails
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if self.transform:
                image = self.transform(image)
            return image, label

        img_h, img_w = image.shape[:2]

        # Convert YOLO coords to pixel coords
        x1 = max(0, int((x_center - w / 2) * img_w))
        y1 = max(0, int((y_center - h / 2) * img_h))
        x2 = min(img_w, int((x_center + w / 2) * img_w))
        y2 = min(img_h, int((y_center + h / 2) * img_h))

        # Crop with a small margin (10%)
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img_w, x2 + margin_x)
        y2 = min(img_h, y2 + margin_y)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.transform:
            crop = self.transform(crop)

        return crop, label


def build_model():
    """Build EfficientNet-B0 binary classifier."""
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    return model


def train_classifier():
    set_seed(RANDOM_SEED)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    print("\nLoading datasets...")
    train_dataset = InsulatorCropDataset('train', transform=train_transform)
    val_dataset = InsulatorCropDataset('val', transform=val_transform)
    test_dataset = InsulatorCropDataset('test', transform=val_transform)

    print(f"  Train: {len(train_dataset)} crops")
    print(f"  Val:   {len(val_dataset)} crops")
    print(f"  Test:  {len(test_dataset)} crops")

    # Count class distribution
    train_labels = [s[5] for s in train_dataset.samples]
    n_normal = sum(1 for l in train_labels if l == 0)
    n_defective = sum(1 for l in train_labels if l == 1)
    print(f"  Train class dist: Normal={n_normal}, Defective={n_defective}")

    # Compute class weights for imbalanced data
    total = n_normal + n_defective
    weight_normal = total / (2 * n_normal) if n_normal > 0 else 1.0
    weight_defective = total / (2 * n_defective) if n_defective > 0 else 1.0
    class_weights = torch.FloatTensor([weight_normal, weight_defective]).to(DEVICE)
    print(f"  Class weights: Normal={weight_normal:.3f}, Defective={weight_defective:.3f}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=False)

    # Model
    model = build_model().to(DEVICE)
    print(f"\nModel: EfficientNet-B0 ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"TRAINING EfficientNet-B0 CLASSIFIER")
    print(f"{'='*60}")

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(MODEL_SAVE_PATH, 'best_classifier.pt'))
            patience_counter = 0
            print(f"  → Best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")

    checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, 'best_classifier.pt'),
                           weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Normal', 'Defective']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved at: {MODEL_SAVE_PATH}/best_classifier.pt")


if __name__ == '__main__':
    train_classifier()
