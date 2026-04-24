"""
Grad-CAM / EigenCAM Visualizer for Insulator Defect Detection
===============================================================
Generates heatmap visualizations showing WHERE the YOLOv8 model
is looking when making predictions.

Uses a hook-based approach: attaches to the YOLOv8 backbone during
normal model.predict() inference, captures activations, and computes
EigenCAM heatmaps.

Usage:
  python gradcam_visualizer.py --image img6.jpg
  python gradcam_visualizer.py --image img6.jpg --conf 0.05
  python gradcam_visualizer.py --image img6.jpg --show

Output:
  Side-by-side image: Original | Detection | Heatmap
  Saved to: ./runs/gradcam/
"""

import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Configuration ---
YOLO_WEIGHTS = './runs/detect/train_v2/weights/best.pt'
OUTPUT_DIR = './runs/gradcam'
DEFAULT_CONF = 0.10


class YOLOEigenCAM:
    """EigenCAM implementation for YOLOv8.

    Hooks into the backbone during model.predict() and captures
    activations, then computes the first principal component (SVD)
    as the attention heatmap.

    Why EigenCAM instead of Grad-CAM for YOLO:
    - Grad-CAM needs class-specific gradients; YOLO's multi-task loss is complex
    - EigenCAM is gradient-free — works directly on activations
    - Better suited for object detectors
    """

    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.activations = None
        self.hook_handle = None

        # Find and hook the target layer
        # YOLOv8 backbone: layer 9 = SPPF (Spatial Pyramid Pooling Fast)
        self._attach_hook()

        print(f"✅ EigenCAM initialized on {device}")

    def _attach_hook(self):
        """Attach a forward hook to the SPPF (backbone output) layer."""
        # Access the internal model layers
        backbone_model = self.model.model.model

        # Find the SPPF layer (typically layer 9 in YOLOv8)
        target_layer = None
        target_name = None

        for i, layer in enumerate(backbone_model):
            layer_type = type(layer).__name__
            if layer_type == 'SPPF':
                target_layer = layer
                target_name = f"Layer {i} ({layer_type})"
                break

        # Fallback: use the last layer before Concat/Detect heads
        if target_layer is None:
            for i, layer in enumerate(backbone_model):
                layer_type = type(layer).__name__
                if layer_type in ('C2f', 'C2fCIB', 'C3'):
                    target_layer = layer
                    target_name = f"Layer {i} ({layer_type})"
            # Use the one we found last (deepest backbone conv)

        if target_layer is None:
            # Ultimate fallback: use layer index 4 (mid-backbone)
            target_layer = backbone_model[4]
            target_name = f"Layer 4 ({type(target_layer).__name__})"

        print(f"   Hooked: {target_name}")

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations = output.detach().cpu()
            elif isinstance(output, (list, tuple)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        self.activations = o.detach().cpu()
                        break

        self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def generate_heatmap(self, image_path, conf=0.10):
        """Run prediction and capture the EigenCAM heatmap.

        Returns:
            heatmap: np.ndarray of shape [H, W] with values in [0, 1]
            result: YOLO prediction result (with boxes)
        """
        # Run prediction — this triggers the hook
        self.activations = None
        results = self.model.predict(
            source=image_path,
            conf=conf,
            device=self.device,
            verbose=False,
            imgsz=640,
        )
        result = results[0]

        # Read original image for sizing
        image = cv2.imread(image_path)
        h_orig, w_orig = image.shape[:2]

        if self.activations is None:
            print("⚠️  No activations captured!")
            return np.zeros((h_orig, w_orig), dtype=np.float32), result

        # EigenCAM computation
        activations = self.activations.squeeze(0)  # [C, H, W]
        C, H, W = activations.shape

        # Reshape to [C, H*W] for SVD
        act_2d = activations.reshape(C, H * W).numpy()

        # SVD: first right singular vector = principal spatial pattern
        try:
            _, _, Vt = np.linalg.svd(act_2d, full_matrices=False)
            principal = Vt[0].reshape(H, W)
        except np.linalg.LinAlgError:
            # Fallback: mean activation across channels
            principal = act_2d.mean(axis=0).reshape(H, W)

        # Normalize to [0, 1]
        principal = np.abs(principal)  # Take absolute value
        principal = principal - principal.min()
        if principal.max() > 0:
            principal = principal / principal.max()

        # Resize to original image dimensions
        heatmap = cv2.resize(principal.astype(np.float32), (w_orig, h_orig))

        return heatmap, result

    def cleanup(self):
        """Remove hooks."""
        if self.hook_handle:
            self.hook_handle.remove()


def apply_heatmap_overlay(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap on an image with transparency."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def draw_detections(image, result):
    """Draw YOLO detections on image with class-specific colors."""
    COLORS = {
        0: (0, 200, 0),      # Green - normal
        1: (0, 0, 255),      # Red - defective
        2: (0, 165, 255),    # Orange - defect region
    }
    LABELS = {
        0: 'Normal Insulator',
        1: 'Defective Insulator',
        2: 'Defect Region',
    }

    annotated = image.copy()
    h, w = annotated.shape[:2]
    thickness = max(2, int(min(w, h) / 300))
    font_scale = max(0.4, min(w, h) / 1500)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS.get(cls_id, (255, 255, 255))
        label = f"{LABELS.get(cls_id, f'Class {cls_id}')} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th_text - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return annotated


def create_side_by_side(original, detection, heatmap_overlay,
                        titles=('Original', 'Detection', 'EigenCAM Heatmap')):
    """Create a 3-panel side-by-side comparison image."""
    h, w = original.shape[:2]

    images = [original, detection, heatmap_overlay]
    resized = []
    for img in images:
        ih, iw = img.shape[:2]
        scale = h / ih
        new_w = int(iw * scale)
        resized.append(cv2.resize(img, (new_w, h)))

    # Add title bars
    titled = []
    for img, title in zip(resized, titles):
        canvas = np.zeros((h + 40, img.shape[1], 3), dtype=np.uint8)
        canvas[40:] = img

        font_scale = max(0.5, img.shape[1] / 600)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        tx = (img.shape[1] - tw) // 2
        cv2.putText(canvas, title, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2)
        titled.append(canvas)

    return np.hstack(titled)


def visualize_gradcam(image_path, conf=0.10, save=True, show=False):
    """Generate and visualize EigenCAM for an image.

    Creates a 3-panel output:
      1. Original image
      2. Detection result (bounding boxes)
      3. EigenCAM heatmap overlay (where the model is looking)
    """
    # Device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"\n{'='*60}")
    print(f"GRAD-CAM / EIGENCAM VISUALIZATION")
    print(f"{'='*60}")
    print(f"Image:  {os.path.basename(image_path)}")
    print(f"Device: {device}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Size:   {w}x{h}")

    # Initialize EigenCAM
    cam = YOLOEigenCAM(YOLO_WEIGHTS, device=device)

    # Generate heatmap + run detection in one pass
    print(f"\n[1/2] Computing EigenCAM + running detection (conf={conf})...")
    heatmap, result = cam.generate_heatmap(image_path, conf=conf)
    n_dets = len(result.boxes)
    print(f"       Found {n_dets} detection(s)")

    # Create visualizations
    print(f"[2/2] Creating visualization...")

    heatmap_overlay = apply_heatmap_overlay(image, heatmap, alpha=0.5)
    detection_img = draw_detections(image, result)
    heatmap_with_boxes = draw_detections(heatmap_overlay, result)
    combined = create_side_by_side(image, detection_img, heatmap_overlay)

    # Save outputs
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        combined_path = os.path.join(OUTPUT_DIR, f"{base_name}_gradcam_combined.jpg")
        cv2.imwrite(combined_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n✅ Combined view: {combined_path}")

        heatmap_path = os.path.join(OUTPUT_DIR, f"{base_name}_heatmap.jpg")
        cv2.imwrite(heatmap_path, heatmap_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✅ Heatmap:       {heatmap_path}")

        overlay_path = os.path.join(OUTPUT_DIR, f"{base_name}_heatmap_detections.jpg")
        cv2.imwrite(overlay_path, heatmap_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✅ Heatmap+boxes: {overlay_path}")

    if show:
        max_display_w = 1800
        if combined.shape[1] > max_display_w:
            scale = max_display_w / combined.shape[1]
            display = cv2.resize(combined, None, fx=scale, fy=scale)
        else:
            display = combined
        cv2.imshow('EigenCAM Visualization', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print interpretation
    print(f"\n{'─'*50}")
    print(f"HOW TO READ THE HEATMAP:")
    print(f"  🔴 Red/Yellow = Model paying MOST attention")
    print(f"  🔵 Blue/Green = Model paying LESS attention")
    print(f"{'─'*50}")
    if n_dets > 0:
        print(f"\n  ✅ {n_dets} detection(s) found.")
        print(f"     Compare heatmap focus with bounding boxes.")
        print(f"     If aligned → model is using correct features.")
        print(f"     If misaligned → model may rely on spurious features.")
    else:
        print(f"\n  ⚠️  No detections found.")
        print(f"     Check where the heatmap focuses:")
        print(f"     - On insulator areas → model sees them but conf too low")
        print(f"     - On background → domain gap issue")

    cam.cleanup()
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM / EigenCAM Visualization for Insulator Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gradcam_visualizer.py --image img6.jpg
  python gradcam_visualizer.py --image img6.jpg --conf 0.05
  python gradcam_visualizer.py --image img6.jpg --show

Output:
  ./runs/gradcam/<image>_gradcam_combined.jpg  — Side-by-side comparison
  ./runs/gradcam/<image>_heatmap.jpg            — Heatmap overlay only
  ./runs/gradcam/<image>_heatmap_detections.jpg — Heatmap + detection boxes
        """
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF,
                        help=f'Detection confidence threshold (default: {DEFAULT_CONF})')
    parser.add_argument('--show', action='store_true',
                        help='Display result in a window')

    args = parser.parse_args()

    visualize_gradcam(
        image_path=args.image,
        conf=args.conf,
        show=args.show,
    )


if __name__ == '__main__':
    main()
