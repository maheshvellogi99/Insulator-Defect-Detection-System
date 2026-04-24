"""
Insulator Dataset Converter v2 — Fixed & Enhanced
===================================================
Converts the CPLID dataset from Pascal VOC XML annotations to YOLO format.

Key improvements over v1:
  1. FIXED: Correctly reads defective insulator labels from labels/insulator/ subdirectory
  2. NEW: Adds defect-region bounding boxes as class 2
  3. NEW: Stratified train/val/test split (70/15/15) with fixed seed
  4. NEW: Verifies all images and labels match
  5. NEW: Prints detailed class distribution statistics

Class mapping:
  0: normal_insulator    — Normal insulator bounding box
  1: defective_insulator — Defective insulator bounding box (whole insulator)
  2: defect_region       — Specific defect area on the insulator
"""

import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import random
from collections import defaultdict

# --- Configuration ---
RANDOM_SEED = 42
ORIGINAL_DATA_PATH = './original_data'
YOLO_OUTPUT_PATH = './yolo_insulator_project/data_v2'

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class IDs
CLASS_NORMAL_INSULATOR = 0
CLASS_DEFECTIVE_INSULATOR = 1
CLASS_DEFECT_REGION = 2

CLASS_NAMES = {
    CLASS_NORMAL_INSULATOR: "normal_insulator",
    CLASS_DEFECTIVE_INSULATOR: "defective_insulator",
    CLASS_DEFECT_REGION: "defect_region",
}


def convert_voc_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax):
    """Convert VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, width, height) normalized."""
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    # Clamp values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def parse_xml_annotation(xml_path):
    """Parse a single VOC XML annotation file and return list of (object_name, bbox) tuples."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        objects.append((name, xmin, ymin, xmax, ymax))

    return objects


def process_normal_insulators(original_base_path):
    """Process Normal_Insulators directory and return list of (image_path, annotations) tuples."""
    img_dir = os.path.join(original_base_path, 'Normal_Insulators', 'images')
    label_dir = os.path.join(original_base_path, 'Normal_Insulators', 'labels')

    samples = []
    xml_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.xml')])

    for xml_file in xml_files:
        img_filename = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found for {xml_file}, skipping")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Could not read image {img_path}, skipping")
            continue

        h, w = img.shape[:2]

        # Parse XML for insulator bounding boxes
        objects = parse_xml_annotation(os.path.join(label_dir, xml_file))

        yolo_annotations = []
        for obj_name, xmin, ymin, xmax, ymax in objects:
            if obj_name == 'insulator':
                bbox = convert_voc_to_yolo(w, h, xmin, ymin, xmax, ymax)
                yolo_annotations.append((CLASS_NORMAL_INSULATOR, *bbox))

        if yolo_annotations:
            samples.append({
                'img_path': img_path,
                'img_filename': img_filename,
                'annotations': yolo_annotations,
                'category': 'normal',
            })

    return samples


def process_defective_insulators(original_base_path):
    """Process Defective_Insulators directory and return list of (image_path, annotations) tuples.

    FIXED: Reads from labels/insulator/ subdirectory (not labels/ root).
    NEW: Also reads defect-region bounding boxes from labels/defect/ subdirectory.
    """
    img_dir = os.path.join(original_base_path, 'Defective_Insulators', 'images')
    insulator_label_dir = os.path.join(original_base_path, 'Defective_Insulators', 'labels', 'insulator')
    defect_label_dir = os.path.join(original_base_path, 'Defective_Insulators', 'labels', 'defect')

    samples = []
    xml_files = sorted([f for f in os.listdir(insulator_label_dir) if f.endswith('.xml')])

    for xml_file in xml_files:
        img_filename = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"  WARNING: Image not found for {xml_file}, skipping")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Could not read image {img_path}, skipping")
            continue

        h, w = img.shape[:2]
        yolo_annotations = []

        # 1. Parse insulator bounding box (whole insulator) → class 1
        insulator_objects = parse_xml_annotation(os.path.join(insulator_label_dir, xml_file))
        for obj_name, xmin, ymin, xmax, ymax in insulator_objects:
            if obj_name == 'insulator':
                bbox = convert_voc_to_yolo(w, h, xmin, ymin, xmax, ymax)
                yolo_annotations.append((CLASS_DEFECTIVE_INSULATOR, *bbox))

        # 2. Parse defect-region bounding box → class 2
        defect_xml_path = os.path.join(defect_label_dir, xml_file)
        if os.path.exists(defect_xml_path):
            defect_objects = parse_xml_annotation(defect_xml_path)
            for obj_name, xmin, ymin, xmax, ymax in defect_objects:
                if obj_name == 'defect':
                    bbox = convert_voc_to_yolo(w, h, xmin, ymin, xmax, ymax)
                    yolo_annotations.append((CLASS_DEFECT_REGION, *bbox))

        if yolo_annotations:
            samples.append({
                'img_path': img_path,
                'img_filename': img_filename,
                'annotations': yolo_annotations,
                'category': 'defective',
            })

    return samples


def stratified_split(normal_samples, defective_samples, seed=42):
    """Perform stratified train/val/test split to ensure class balance in each split."""
    random.seed(seed)

    def split_list(items):
        random.shuffle(items)
        n = len(items)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        return items[:train_end], items[train_end:val_end], items[val_end:]

    normal_train, normal_val, normal_test = split_list(normal_samples.copy())
    defective_train, defective_val, defective_test = split_list(defective_samples.copy())

    train = normal_train + defective_train
    val = normal_val + defective_val
    test = normal_test + defective_test

    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_split(samples, split_name, output_base):
    """Write samples to YOLO format directory structure."""
    img_out = os.path.join(output_base, 'images', split_name)
    lbl_out = os.path.join(output_base, 'labels', split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for sample in samples:
        # Copy image
        dst_img = os.path.join(img_out, sample['img_filename'])
        shutil.copy2(sample['img_path'], dst_img)

        # Write label
        txt_filename = sample['img_filename'].replace('.jpg', '.txt')
        txt_path = os.path.join(lbl_out, txt_filename)
        with open(txt_path, 'w') as f:
            for ann in sample['annotations']:
                cls_id, x, y, w, h = ann
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def print_statistics(train, val, test):
    """Print detailed dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for split_name, samples in [("Train", train), ("Val", val), ("Test", test)]:
        class_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for sample in samples:
            category_counts[sample['category']] += 1
            for ann in sample['annotations']:
                class_counts[ann[0]] += 1

        print(f"\n{split_name} split: {len(samples)} images")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat} images: {count}")
        for cls_id in sorted(class_counts.keys()):
            print(f"  Class {cls_id} ({CLASS_NAMES[cls_id]}): {class_counts[cls_id]} annotations")

    print("\n" + "=" * 60)


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("=" * 60)
    print("CPLID → YOLO Converter v2 (FIXED)")
    print("=" * 60)

    # Clean output directory
    if os.path.exists(YOLO_OUTPUT_PATH):
        print(f"\nRemoving existing output directory: {YOLO_OUTPUT_PATH}")
        shutil.rmtree(YOLO_OUTPUT_PATH)

    # Process Normal Insulators
    print("\n[1/4] Processing Normal Insulators (Class 0)...")
    normal_samples = process_normal_insulators(ORIGINAL_DATA_PATH)
    print(f"  Found {len(normal_samples)} normal insulator images")

    # Process Defective Insulators
    print("\n[2/4] Processing Defective Insulators (Class 1 + Class 2)...")
    defective_samples = process_defective_insulators(ORIGINAL_DATA_PATH)
    print(f"  Found {len(defective_samples)} defective insulator images")

    # Stratified split
    print(f"\n[3/4] Splitting dataset (seed={RANDOM_SEED})...")
    print(f"  Ratios: train={TRAIN_RATIO:.0%}, val={VAL_RATIO:.0%}, test={TEST_RATIO:.0%}")
    train, val, test = stratified_split(normal_samples, defective_samples, seed=RANDOM_SEED)

    # Write to disk
    print(f"\n[4/4] Writing YOLO format data to {YOLO_OUTPUT_PATH}...")
    write_split(train, 'train', YOLO_OUTPUT_PATH)
    write_split(val, 'val', YOLO_OUTPUT_PATH)
    write_split(test, 'test', YOLO_OUTPUT_PATH)

    # Statistics
    print_statistics(train, val, test)

    print("\n✅ Dataset conversion complete!")
    print(f"   Output directory: {YOLO_OUTPUT_PATH}")
