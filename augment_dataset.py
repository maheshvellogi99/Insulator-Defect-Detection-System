"""
Offline Data Augmentation for Insulator Defect Detection
=========================================================
Expands the training set ~5x with focus on:
  1. Oversampling defective images (class balance)
  2. Geometric transforms (rotation, flip, crop, affine)
  3. Photometric transforms (brightness, contrast, noise, blur)
  4. Domain-specific transforms (weather simulation)

Uses albumentations library with proper YOLO bbox handling.
"""

import os
import cv2
import numpy as np
import random
import albumentations as A
from copy import deepcopy

# --- Configuration ---
RANDOM_SEED = 42
DATA_PATH = './yolo_insulator_project/data_v2'
TRAIN_IMAGES = os.path.join(DATA_PATH, 'images', 'train')
TRAIN_LABELS = os.path.join(DATA_PATH, 'labels', 'train')

# How many augmented copies per image
NORMAL_AUG_COPIES = 3      # 3x for normal (already have more)
DEFECTIVE_AUG_COPIES = 8   # 8x for defective (minority class — oversample)


def create_augmentation_pipeline():
    """Create a robust augmentation pipeline using albumentations."""
    transform = A.Compose([
        # Geometric transforms
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.7),

        A.OneOf([
            A.Affine(
                rotate=(-45, 45),
                scale=(0.8, 1.2),
                shear=(-10, 10),
                mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.Perspective(scale=(0.02, 0.08), p=1.0),
        ], p=0.5),

        A.OneOf([
            A.RandomCrop(height=480, width=480, p=1.0),
            A.CenterCrop(height=512, width=512, p=1.0),
        ], p=0.2),

        # Always resize back to a standard size to keep consistency
        A.Resize(height=640, width=640, p=1.0),

        # Photometric transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=1.0
            ),
        ], p=0.7),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.3),

        # Weather/domain simulation
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            A.RandomShadow(p=1.0),
            A.RandomSunFlare(
                src_radius=100,
                p=1.0,
            ),
        ], p=0.2),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,  # Drop bboxes that become too occluded
        min_area=100,        # Drop tiny bboxes
    ))

    return transform


def read_yolo_labels(label_path):
    """Read YOLO format label file and return bboxes and class labels."""
    bboxes = []
    class_labels = []

    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                # Clamp to valid range
                x_center = max(0.001, min(0.999, x_center))
                y_center = max(0.001, min(0.999, y_center))
                w = max(0.001, min(0.999, w))
                h = max(0.001, min(0.999, h))
                # Ensure bbox doesn't extend beyond image
                if x_center - w / 2 < 0:
                    w = x_center * 2
                if y_center - h / 2 < 0:
                    h = y_center * 2
                if x_center + w / 2 > 1:
                    w = (1 - x_center) * 2
                if y_center + h / 2 > 1:
                    h = (1 - y_center) * 2
                bboxes.append([x_center, y_center, w, h])
                class_labels.append(cls_id)

    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    """Write YOLO format label file."""
    with open(label_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            x, y, w, h = bbox
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def is_defective_image(label_path):
    """Check if an image contains defective insulator annotations (class 1 or 2)."""
    _, class_labels = read_yolo_labels(label_path)
    return any(c in [1, 2] for c in class_labels)


def augment_dataset():
    """Apply offline augmentation to the training set."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    transform = create_augmentation_pipeline()

    image_files = sorted([f for f in os.listdir(TRAIN_IMAGES) if f.endswith('.jpg')])
    total_original = len(image_files)

    print(f"Found {total_original} original training images")

    augmented_count = 0
    skipped_count = 0

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(TRAIN_IMAGES, img_file)
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(TRAIN_LABELS, label_file)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  WARNING: Could not read {img_path}")
            continue

        # Read labels
        bboxes, class_labels = read_yolo_labels(label_path)
        if not bboxes:
            continue

        # Determine number of augmentation copies
        is_defective = is_defective_image(label_path)
        num_copies = DEFECTIVE_AUG_COPIES if is_defective else NORMAL_AUG_COPIES

        for copy_idx in range(num_copies):
            try:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )

                new_image = transformed['image']
                new_bboxes = transformed['bboxes']
                new_class_labels = transformed['class_labels']

                # Skip if all bboxes were lost during augmentation
                if not new_bboxes:
                    skipped_count += 1
                    continue

                # Generate unique filename
                base_name = img_file.replace('.jpg', '')
                aug_img_file = f"{base_name}_aug{copy_idx}.jpg"
                aug_label_file = f"{base_name}_aug{copy_idx}.txt"

                # Save augmented image and labels
                cv2.imwrite(
                    os.path.join(TRAIN_IMAGES, aug_img_file),
                    new_image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                write_yolo_labels(
                    os.path.join(TRAIN_LABELS, aug_label_file),
                    new_bboxes,
                    new_class_labels,
                )
                augmented_count += 1

            except Exception as e:
                skipped_count += 1
                continue

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total_original} images, "
                  f"generated {augmented_count} augmented images...")

    print(f"\n✅ Augmentation complete!")
    print(f"  Original images: {total_original}")
    print(f"  Augmented images generated: {augmented_count}")
    print(f"  Skipped (bbox lost): {skipped_count}")
    print(f"  Total training images: {total_original + augmented_count}")

    # Print new class distribution
    print("\n  New class distribution in training set:")
    all_labels = sorted([f for f in os.listdir(TRAIN_LABELS) if f.endswith('.txt')])
    class_counts = {0: 0, 1: 0, 2: 0}
    for lf in all_labels:
        _, cls_labels = read_yolo_labels(os.path.join(TRAIN_LABELS, lf))
        for c in cls_labels:
            class_counts[c] = class_counts.get(c, 0) + 1
    for cls_id, count in sorted(class_counts.items()):
        names = {0: "normal_insulator", 1: "defective_insulator", 2: "defect_region"}
        print(f"    Class {cls_id} ({names.get(cls_id, 'unknown')}): {count}")


if __name__ == '__main__':
    print("=" * 60)
    print("OFFLINE DATA AUGMENTATION")
    print("=" * 60)
    augment_dataset()
