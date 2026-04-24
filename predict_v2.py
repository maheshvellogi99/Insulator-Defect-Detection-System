"""
Unified Prediction Script v2 (Enhanced)
=========================================
Two-stage insulator defect detection with multi-scale inference:
  Stage 1: YOLOv8m detects insulators and defect regions
  Stage 2: EfficientNet-B0 classifies cropped insulators as Normal/Defective

Key improvements:
  - Multi-scale SAHI-style tiled inference for wide-angle images
  - Lower default confidence for real-world images
  - Diagnostic mode showing all raw detections
  - Better visualization with class-specific colors and labels
  - Auto-detects if image needs tiling (large images with small objects)

Usage:
  python predict_v2.py --image path/to/image.jpg
  python predict_v2.py --image path/to/image.jpg --yolo-only
  python predict_v2.py --image path/to/image.jpg --conf 0.05 --verbose
  python predict_v2.py --image path/to/image.jpg --no-sahi
"""

import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import timm
from torchvision import transforms

# --- Configuration ---
YOLO_WEIGHTS = './runs/detect/train_v2/weights/best.pt'
CLASSIFIER_WEIGHTS = './runs/classifier/best_classifier.pt'
OUTPUT_DIR = './runs/predictions_v2'
IMG_SIZE_CLASSIFIER = 224

# Detection thresholds — lowered for real-world images
YOLO_CONF = 0.10       # Was 0.25 — too high for out-of-distribution images
YOLO_IOU = 0.45        # NMS IoU threshold

# SAHI tiling parameters
SAHI_TILE_SIZE = 640    # Tile size in pixels
SAHI_OVERLAP = 0.25     # 25% overlap between tiles
SAHI_MIN_IMAGE_DIM = 800  # Auto-enable SAHI if image dimension > this

# Colors for visualization (BGR)
COLORS = {
    'normal_insulator': (0, 200, 0),       # Green — normal, safe
    'defective_insulator': (0, 0, 255),     # Red — defective, danger
    'defect_region': (0, 165, 255),          # Orange — specific defect area
    'normal_classified': (0, 200, 0),        # Green
    'defective_classified': (0, 0, 255),     # Red
}

# Class labels with display names
CLASS_NAMES_YOLO = {
    0: 'normal_insulator',
    1: 'defective_insulator',
    2: 'defect_region',
}

DISPLAY_NAMES = {
    'normal_insulator': 'Normal Insulator',
    'defective_insulator': 'Defective Insulator',
    'defect_region': 'Defect Region',
}


def load_classifier():
    """Load the trained EfficientNet-B0 classifier."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

    if os.path.exists(CLASSIFIER_WEIGHTS):
        checkpoint = torch.load(CLASSIFIER_WEIGHTS, weights_only=True,
                               map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded classifier from {CLASSIFIER_WEIGHTS}")
    else:
        print(f"⚠️  Classifier weights not found at {CLASSIFIER_WEIGHTS}")
        print(f"   Running in YOLO-only mode (Stage 2 classification disabled)")
        return None

    model.eval()
    return model


def classify_crop(model, crop_bgr, device='cpu'):
    """Classify a cropped insulator region as Normal/Defective."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = 'Normal' if predicted.item() == 0 else 'Defective'
    return class_name, confidence.item()


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def nms_across_tiles(detections, iou_threshold=0.5):
    """Apply NMS across all tiled detections to remove duplicates."""
    if not detections:
        return []

    # Sort by confidence (descending)
    detections.sort(key=lambda d: d['conf'], reverse=True)

    keep = []
    for det in detections:
        is_duplicate = False
        for kept in keep:
            if det['cls_id'] == kept['cls_id']:
                iou = compute_iou(det['box'], kept['box'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            keep.append(det)

    return keep


def run_yolo_on_region(yolo_model, image, offset_x, offset_y, conf, iou, device):
    """Run YOLO on a region of the image and return detections in global coords."""
    results = yolo_model.predict(
        source=image,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    detections = []
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Convert to global coordinates
        detections.append({
            'cls_id': cls_id,
            'conf': confidence,
            'box': [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
        })

    return detections


def predict_with_sahi(yolo_model, image, conf, iou, device,
                      tile_size=640, overlap=0.25):
    """Run SAHI-style tiled inference for detecting small objects in large images."""
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))

    all_detections = []

    # 1. Run on FULL image (catches large objects)
    print(f"  [SAHI] Running on full image ({w}x{h})...")
    full_dets = run_yolo_on_region(yolo_model, image, 0, 0, conf, iou, device)
    all_detections.extend(full_dets)
    print(f"         → {len(full_dets)} detections")

    # 2. Run on TILES (catches small objects)
    tile_count = 0
    tile_dets_total = 0

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)

            # Skip tiny edge tiles
            if (x1 - x0) < tile_size // 3 or (y1 - y0) < tile_size // 3:
                continue

            tile = image[y0:y1, x0:x1]
            tile_count += 1

            tile_dets = run_yolo_on_region(
                yolo_model, tile, x0, y0, conf, iou, device
            )
            tile_dets_total += len(tile_dets)
            all_detections.extend(tile_dets)

    print(f"  [SAHI] Ran on {tile_count} tiles → {tile_dets_total} detections")

    # 3. Run on 2x UPSCALED image (catches very small objects)
    if max(h, w) < 2000:
        scale = 2.0
        upscaled = cv2.resize(image, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)
        print(f"  [SAHI] Running on 2x upscaled ({int(w*scale)}x{int(h*scale)})...")
        upscaled_dets = run_yolo_on_region(
            yolo_model, upscaled, 0, 0, conf, iou, device
        )
        # Scale detections back to original coordinates
        for det in upscaled_dets:
            det['box'] = [int(c / scale) for c in det['box']]
        all_detections.extend(upscaled_dets)
        print(f"         → {len(upscaled_dets)} detections")

    # 4. NMS across ALL detections
    merged = nms_across_tiles(all_detections, iou_threshold=YOLO_IOU)
    print(f"  [SAHI] After NMS: {len(merged)} unique detections")

    return merged


def draw_detection(result_image, x1, y1, x2, y2, label, confidence, color):
    """Draw a single detection box with label on the image."""
    h, w = result_image.shape[:2]

    # Adaptive thickness and font
    thickness = max(2, int(min(w, h) / 300))
    font_scale = max(0.5, min(w, h) / 1200)

    # Draw bounding box
    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

    # Draw label background + text
    label_text = f"{label} {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Position label above box, or below if at top edge
    label_y = y1 - 5 if y1 > th + 15 else y2 + th + 10

    if label_y == y1 - 5:
        cv2.rectangle(result_image, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    else:
        cv2.rectangle(result_image, (x1, y2), (x1 + tw + 6, y2 + th + 10), color, -1)

    cv2.putText(result_image, label_text, (x1 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


def predict(image_path, yolo_only=False, save=True, show=False,
            conf=None, use_sahi=True, verbose=False):
    """Run two-stage prediction on a single image.

    The 3-class detection works as follows:
      - Class 0 (normal_insulator): Green box around healthy insulators
      - Class 1 (defective_insulator): Red box around the whole defective insulator
      - Class 2 (defect_region): Orange box pinpointing the specific defect area

    For defective insulators, you'll see BOTH a red box (whole insulator)
    AND an orange box (defect area within it).
    """
    detection_conf = conf if conf is not None else YOLO_CONF

    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"\n{'='*60}")
    print(f"INSULATOR DEFECT DETECTION v2")
    print(f"{'='*60}")
    print(f"Image:  {os.path.basename(image_path)}")
    print(f"Device: {device}")
    print(f"Conf:   {detection_conf}")
    print(f"SAHI:   {'enabled' if use_sahi else 'disabled'}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image {image_path}")
        return

    result_image = image.copy()
    h, w = image.shape[:2]
    print(f"Size:   {w}x{h}")

    # Load YOLO model
    print(f"\n[Stage 1] YOLOv8 Detection...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    # Decide inference strategy
    should_sahi = use_sahi and (max(h, w) > SAHI_MIN_IMAGE_DIM)

    if should_sahi:
        print(f"  → Using SAHI tiled inference (image is {max(h,w)}px > {SAHI_MIN_IMAGE_DIM}px)")
        raw_detections = predict_with_sahi(
            yolo_model, image, detection_conf, YOLO_IOU, device,
            tile_size=SAHI_TILE_SIZE, overlap=SAHI_OVERLAP
        )
    else:
        # Standard single-pass inference
        print(f"  → Standard inference...")
        results = yolo_model.predict(
            source=image_path,
            conf=detection_conf,
            iou=YOLO_IOU,
            device=device,
            verbose=False,
            imgsz=640,
        )

        raw_detections = []
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_detections.append({
                'cls_id': cls_id,
                'conf': confidence,
                'box': [x1, y1, x2, y2],
            })

        # Also try on upscaled version for small objects
        if not raw_detections and max(h, w) <= SAHI_MIN_IMAGE_DIM:
            print(f"  → No detections! Retrying with 2x upscale...")
            scale = 2.0
            upscaled = cv2.resize(image, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LINEAR)
            results2 = yolo_model.predict(
                source=upscaled,
                conf=detection_conf * 0.5,  # Even lower conf for retry
                iou=YOLO_IOU,
                device=device,
                verbose=False,
                imgsz=640,
            )
            for box in results2[0].boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_detections.append({
                    'cls_id': cls_id,
                    'conf': confidence,
                    'box': [int(x1/scale), int(y1/scale),
                            int(x2/scale), int(y2/scale)],
                })
            if raw_detections:
                raw_detections = nms_across_tiles(raw_detections, YOLO_IOU)
                print(f"  → Found {len(raw_detections)} detections with upscale!")

    print(f"\n  Total detections: {len(raw_detections)}")

    if not raw_detections:
        print(f"\n⚠️  No detections found!")
        print(f"   Try: --conf 0.05 (lower threshold)")
        print(f"   Or the image may be too different from training data")

    # Stage 2: Classifier (optional)
    classifier = None
    if not yolo_only:
        classifier = load_classifier()
        if classifier is not None:
            classifier = classifier.to(device)

    # Track detection counts by class
    class_counts = {0: 0, 1: 0, 2: 0}

    # Process each detection
    for i, det in enumerate(raw_detections):
        cls_id = det['cls_id']
        yolo_conf = det['conf']
        x1, y1, x2, y2 = det['box']
        class_name = CLASS_NAMES_YOLO.get(cls_id, f'class_{cls_id}')

        # Clamp coordinates to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        if verbose:
            print(f"\n  Detection {i+1}:")
            print(f"    YOLO: {class_name} (conf={yolo_conf:.3f})")
            print(f"    Box: ({x1}, {y1}) → ({x2}, {y2})")

        # Determine final label and color
        display_name = DISPLAY_NAMES.get(class_name, class_name)
        final_label = display_name
        final_conf = yolo_conf
        color = COLORS.get(class_name, (255, 255, 255))

        # If it's an insulator (class 0 or 1) and we have a classifier — use Stage 2
        if classifier is not None and cls_id in [0, 1]:
            # Crop insulator region with margin
            margin_x = int((x2 - x1) * 0.05)
            margin_y = int((y2 - y1) * 0.05)
            cx1 = max(0, x1 - margin_x)
            cy1 = max(0, y1 - margin_y)
            cx2 = min(w, x2 + margin_x)
            cy2 = min(h, y2 + margin_y)

            crop = image[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                cls_name, cls_conf = classify_crop(classifier, crop, device)
                if verbose:
                    print(f"    Classifier: {cls_name} (conf={cls_conf:.3f})")

                final_label = f"{cls_name} Insulator"
                final_conf = (yolo_conf + cls_conf) / 2
                color = COLORS.get(f'{cls_name.lower()}_classified', (255, 255, 255))

        elif cls_id == 2:
            # Defect Region — keep YOLO label
            final_label = "Defect Region"
            color = COLORS['defect_region']

        # Draw on result image
        draw_detection(result_image, x1, y1, x2, y2, final_label, final_conf, color)

    # Print summary
    print(f"\n{'─'*40}")
    print(f"Detection Summary:")
    print(f"  🟢 Normal Insulators:    {class_counts.get(0, 0)}")
    print(f"  🔴 Defective Insulators: {class_counts.get(1, 0)}")
    print(f"  🟠 Defect Regions:       {class_counts.get(2, 0)}")
    print(f"{'─'*40}")

    # Add legend to image
    legend_y = 30
    font_scale = max(0.5, min(w, h) / 1500)
    for label, color in [('Normal Insulator', COLORS['normal_insulator']),
                          ('Defective Insulator', COLORS['defective_insulator']),
                          ('Defect Region', COLORS['defect_region'])]:
        cv2.rectangle(result_image, (10, legend_y - 15), (30, legend_y + 5), color, -1)
        cv2.putText(result_image, label, (35, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        legend_y += 30

    # Save result
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_prediction.jpg")
        cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n✅ Saved prediction to: {output_path}")

    if show:
        cv2.imshow('Insulator Defect Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_image


def main():
    parser = argparse.ArgumentParser(
        description='Insulator Defect Detection v2 — Multi-scale Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_v2.py --image img6.jpg
  python predict_v2.py --image img6.jpg --conf 0.05 --verbose
  python predict_v2.py --image img3.jpg --yolo-only
  python predict_v2.py --image img6.jpg --no-sahi

Detection Classes:
  🟢 normal_insulator    — Healthy insulator (green box)
  🔴 defective_insulator — Damaged insulator (red box around whole insulator)
  🟠 defect_region       — Specific defect area (orange box within insulator)
        """
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--yolo-only', action='store_true',
                        help='Use only YOLO detection (skip EfficientNet classifier)')
    parser.add_argument('--show', action='store_true',
                        help='Display result in a window')
    parser.add_argument('--conf', type=float, default=YOLO_CONF,
                        help=f'Detection confidence threshold (default: {YOLO_CONF})')
    parser.add_argument('--no-sahi', action='store_true',
                        help='Disable SAHI tiled inference')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed info for each detection')

    args = parser.parse_args()

    predict(
        image_path=args.image,
        yolo_only=args.yolo_only,
        show=args.show,
        conf=args.conf,
        use_sahi=not args.no_sahi,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
