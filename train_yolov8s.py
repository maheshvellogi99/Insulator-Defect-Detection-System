"""
YOLOv8s Training Script for Insulator Defect Detection
=======================================================
Trains YOLOv8s (small) with optimized hyperparameters on Apple M2 GPU (MPS).

Key improvements over previous YOLOv8n training:
  - 3.5x larger model (11.2M vs 3.2M params)
  - 3 classes with proper labels (FIXED: defective labels now included!)
  - Heavy online augmentation (mosaic, mixup, copy_paste)
  - Cosine LR schedule
  - Early stopping with patience=20
  - AMP DISABLED — MPS + AMP causes NaN gradients (known PyTorch bug)
  
Note: YOLOv8m (25.9M) causes NaN losses on MPS with AMP. 
      YOLOv8s is the best balance of accuracy vs stability on M2 8GB.
"""

from ultralytics import YOLO
import torch
import os

def main():
    print("=" * 60)
    print("YOLOv8s TRAINING — Insulator Defect Detection v2")
    print("=" * 60)

    # Check device — MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"✅ Using Apple M2 GPU (MPS)")
        print(f"⚠️  AMP disabled (MPS + AMP = NaN gradients)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"⚠️  Using CPU (training will be slow)")

    # Load YOLOv8s pretrained on COCO
    model = YOLO('yolov8s.pt')
    print(f"✅ Loaded YOLOv8s ({sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M params)")

    # Determine if we should use AMP
    use_amp = (device != 'mps')  # Disable AMP on MPS

    # Train with optimized hyperparameters
    results = model.train(
        # Data
        data='dataset_v2.yaml',
        imgsz=640,
        
        # Training schedule
        epochs=100,
        patience=20,         # Early stopping
        batch=16,            # Can use 16 with YOLOv8s (smaller model)
        
        # Device
        device=device,
        workers=4,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,           # Lower initial LR for AdamW
        lrf=0.01,            # Final LR ratio
        cos_lr=True,         # Cosine annealing
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,
        
        # Loss weights
        box=7.5,
        cls=1.5,             # Increased from 0.5 — classification is key
        dfl=1.5,
        
        # Online augmentation (heavy)
        mosaic=1.0,
        mixup=0.3,           # Was 0.0
        copy_paste=0.3,      # Was 0.0
        degrees=45,          # Was 0.0
        translate=0.2,       # Was 0.1
        scale=0.5,
        shear=10.0,          # Was 0.0
        perspective=0.001,   # Was 0.0
        flipud=0.5,          # Was 0.0
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        
        # Close mosaic for last 15 epochs (stabilize)
        close_mosaic=15,
        
        # Misc
        seed=42,
        deterministic=True,
        pretrained=True,
        save=True,
        save_period=10,       # Save checkpoint every 10 epochs
        plots=True,
        name='train_v2',
        exist_ok=True,
        
        # AMP DISABLED on MPS (causes NaN)
        amp=use_amp,
        
        # Multi-scale training for robustness
        multi_scale=False,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    # Validate on test set
    print("\n[Evaluating on TEST set...]")
    best_model = YOLO('./runs/detect/train_v2/weights/best.pt')
    test_results = best_model.val(
        data='dataset_v2.yaml',
        split='test',
        device=device,
        batch=8,
        imgsz=640,
        plots=True,
        name='test_eval_v2',
        exist_ok=True,
    )

    print(f"\nTest Set Results:")
    print(f"  mAP50:    {test_results.box.map50:.4f}")
    print(f"  mAP50-95: {test_results.box.map:.4f}")

    return results


if __name__ == '__main__':
    main()
