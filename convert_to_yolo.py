import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import random

# YOLO classes: 0 for Normal, 1 for Defective
CLASSES = {"normal": 0, "defective": 1}

def convert_coordinates(size, box):
    # Convert VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, width, height) normalized to 0-1
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_dataset(original_base_path, category_name, class_id, output_base):
    img_dir = os.path.join(original_base_path, category_name, 'images')
    label_dir = os.path.join(original_base_path, category_name, 'labels')
    
    # Get all XML files and shuffle them for a random train/val split
    xml_files = [f for f in os.listdir(label_dir) if f.endswith('.xml')]
    random.shuffle(xml_files)
    
    # 80% for training, 20% for validation
    split_index = int(len(xml_files) * 0.8)
    train_files = xml_files[:split_index]
    
    for xml_file in xml_files:
        subset = 'train' if xml_file in train_files else 'val'
        
        # Parse XML
        tree = ET.parse(os.path.join(label_dir, xml_file))
        root = tree.getroot()
        
        # Get image dimensions
        img_filename = xml_file.replace('.xml', '.jpg')
        img_path = os.path.join(img_dir, img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Prepare YOLO text file
        txt_filename = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(output_base, 'labels', subset, txt_filename)
        
        with open(txt_path, 'w') as out_file:
            for obj in root.findall('object'):
                name = obj.find('name').text
                # We only want the main insulator bounding box
                if name == 'insulator':
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_coordinates((w, h), b)
                    # Write: class_id x_center y_center width height
                    out_file.write(f"{class_id} " + " ".join([str(a) for a in bb]) + '\n')
        
        # Copy the image to the YOLO images folder
        shutil.copy(img_path, os.path.join(output_base, 'images', subset, img_filename))

# --- EXECUTION ---
# Set this to where you extracted the CPLID dataset
ORIGINAL_DATA_PATH = './original_data' 
YOLO_OUTPUT_PATH = './yolo_insulator_project/data'

# Create necessary directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_OUTPUT_PATH, 'labels', split), exist_ok=True)

print("Processing Normal Insulators (Class 0)...")
process_dataset(ORIGINAL_DATA_PATH, 'Normal_Insulators', CLASSES["normal"], YOLO_OUTPUT_PATH)

print("Processing Defective Insulators (Class 1)...")
process_dataset(ORIGINAL_DATA_PATH, 'Defective_Insulators', CLASSES["defective"], YOLO_OUTPUT_PATH)

print("Dataset successfully converted to YOLO format!")