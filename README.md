# Insulator Defect Detection using YOLOv8

## Project overview
This repository trains and uses a YOLOv8 object detector to find insulators in images and classify each detected insulator as one of:

1. `normal_insulator`
2. `defective_insulator`

The repo includes:

- A dataset conversion script that converts Pascal VOC XML annotations into YOLO label format.
- A YOLOv8 training run with saved artifacts (weights, curves, confusion matrix, and per-epoch metrics).
- A small inference script that loads the best weights and runs prediction on a sample image.

## What problem is being solved?
Given an input image containing power line insulators, the model predicts bounding boxes around insulators and assigns each bounding box a class label (`normal_insulator` or `defective_insulator`).

This is an *object detection* problem (not a pure image classification problem), because the output includes both:

- **Localization**: where the insulator is (bounding box)
- **Classification**: whether the detected insulator is normal or defective

## Approach used
### Model
- **Framework**: Ultralytics YOLO (YOLOv8)
- **Starting weights**: `yolov8n.pt` (YOLOv8 Nano, pretrained on COCO)
- **Training strategy**: transfer learning / fine-tuning on the insulator dataset

YOLOv8 is a single-stage detector that predicts boxes and class probabilities in one forward pass. In Ultralytics YOLOv8, the loss is a combination of:

- **Box loss** (localization)
- **Cls loss** (classification)
- **DFL loss** (distribution focal loss used in modern YOLO box regression)

### Data preparation
The raw dataset is stored under `original_data/` and contains Pascal VOC-style XML annotations.

The script `convert_to_yolo.py`:

- Reads XML annotations.
- Extracts the bounding box for the object named `insulator`.
- Converts VOC `(xmin, ymin, xmax, ymax)` to YOLO normalized `(x_center, y_center, width, height)`.
- Maps dataset category to class id:
  - `normal` -> `0`
  - `defective` -> `1`
- Performs a random **80/20 split** into `train` and `val`.
- Writes YOLO `.txt` label files and copies images into YOLO folder structure.

## Repository structure (what every folder/file is)
### Top-level
- **`.venv/`**
  - Local Python virtual environment. Not required for the project to work, but commonly used during development.

- **`original_data/`**
  - Raw dataset extracted locally.
  - Expected structure:
    - `original_data/Normal_Insulators/images/` and `original_data/Normal_Insulators/labels/`
    - `original_data/Defective_Insulators/images/` and `original_data/Defective_Insulators/labels/`
  - In this workspace snapshot, the `labels/` folders are present (XMLs), but the `images/` folders appear empty. If images are not present locally, conversion/training will not be reproducible.

- **`yolo_insulator_project/`**
  - Generated YOLO dataset folder.
  - `yolo_insulator_project/data/` contains YOLO-style `images/` and `labels/` split into `train/` and `val/`.
  - In this workspace snapshot, `labels/` are present, but `images/` directories appear empty. Ultralytics training requires images; if you want to retrain, ensure the images exist and paths are correct.

- **`dataset.yaml`**
  - Dataset configuration used by Ultralytics.
  - Defines:
    - `path`: dataset root
    - `train`: relative path to training images
    - `val`: relative path to validation images
    - `names`: class id to class name mapping
  - Important: the current `path` is hard-coded to a different machine path (`/Users/hari/...`). You will likely need to update this to your local path.

- **`convert_to_yolo.py`**
  - Conversion script described above.
  - Produces output under `./yolo_insulator_project/data`.

- **`yolov8n.pt`**
  - Pretrained YOLOv8 Nano checkpoint used for transfer learning.

- **`predict.py`**
  - Minimal inference script:
    - Loads best weights from `./runs/detect/train2/weights/best.pt`
    - Runs prediction on `img6.jpg`
    - Saves prediction output under `runs/detect/predict*/`
    - Uses `conf=0.1` to visualize low-confidence detections

- **`runs/`**
  - Ultralytics-generated run artifacts.
  - Contains:
    - `runs/detect/train2/` (a completed training run with metrics, plots, and weights)
    - `runs/detect/train/` and `runs/detect/train3/` (incomplete/partial runs in this snapshot)
    - `runs/detect/predict*/` (inference output images)

### `runs/detect/train2/` artifacts
This directory is the key “results” directory.

- **`weights/best.pt`**: best checkpoint selected by Ultralytics on validation metrics.
- **`weights/last.pt`**: final checkpoint at end of training.
- **`args.yaml`**: the exact training hyperparameters used.
- **`results.csv`**: per-epoch metrics and losses.
- **`results.png`**: plot of key losses/metrics over epochs.
- **`confusion_matrix*.png`**: confusion matrices.
- **`BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png`**: precision/recall/F1/PR curves.
- **`train_batch*.jpg`, `val_batch*_pred.jpg`**: qualitative examples of training and validation predictions.

## How the model was trained (reproducibility)
The saved run indicates YOLOv8 training with:

- **Task**: detect
- **Base model**: `yolov8n.pt`
- **Epochs**: 50
- **Batch size**: 16
- **Image size**: 640
- **Device**: CPU (per `runs/detect/train2/args.yaml`)

You can retrain with Ultralytics either via CLI or Python.

### Option A: CLI (recommended)
1. Install Ultralytics (see “Environment setup”).
2. Ensure `dataset.yaml` points to your local dataset.
3. Run:

```bash
yolo detect train model=yolov8n.pt data=dataset.yaml imgsz=640 batch=16 epochs=50
```

### Option B: Python
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', imgsz=640, batch=16, epochs=50)
```

## Best results achieved in this repo
The best metrics below are taken from `runs/detect/train2/results.csv`.

- **Best `metrics/mAP50(B)`**: `0.97963` (epoch 45)
- **Best `metrics/mAP50-95(B)`**: `0.85818` (epoch 48)
- **Best `metrics/precision(B)`**: `0.96037` (epoch 1)
- **Best `metrics/recall(B)`**: `0.97297` (epoch 43)
- **Final epoch (50) validation losses**:
  - `val/box_loss`: `0.56676`
  - `val/cls_loss`: `0.41821`
  - `val/dfl_loss`: `0.90222`

Notes for interpreting these numbers:

- **mAP50** is usually easier/higher than **mAP50-95** (which is stricter and more representative of precise localization).
- Precision being highest at epoch 1 suggests the dataset/validation set may be small, or the metric can fluctuate; it’s more reliable to look at mAP trends and PR/F1 curves.

## How to run inference (prediction)
The repository includes `predict.py`:

```bash
python predict.py
```

It loads:

- `./runs/detect/train2/weights/best.pt`

and runs inference on:

- `img6.jpg`

Ultralytics will write output images to:

- `runs/detect/predict*/`

## Environment setup
This repo does not include `requirements.txt` / `pyproject.toml`.

Minimum needed to run training/inference (conceptually):

- `ultralytics`
- `torch`
- `opencv-python` (used by `convert_to_yolo.py`)

If you are using the included `.venv`, ensure you are running Python from that environment.

## Common issues / repo-specific caveats
1. **`dataset.yaml` hard-coded path**
   - Current value: `path: /Users/hari/insulator_project/yolo_insulator_project/data`
   - You likely need to change it to: `path: /Users/maheshvellogi/Downloads/Insulator_Project/yolo_insulator_project/data` (or make it relative).

2. **Images appear missing in dataset folders**
   - `yolo_insulator_project/data/images/train` and `.../val` are empty in this workspace snapshot.
   - `original_data/*/images` are also empty here.
   - If you want full reproducibility, ensure the original JPG images exist locally and rerun `convert_to_yolo.py`.

3. **Train run portability**
   - `runs/detect/train2/args.yaml` has `save_dir: /Users/hari/Insulator_Project/...` which is just metadata from the machine where training happened.

## How to improve the model (practical next steps)
If your goal is to improve real-world performance (not just validation on a small split), prioritize these:

1. **Fix dataset paths and ensure images are present**
   - Without the full dataset, improvements can’t be validated reliably.

2. **Use a deterministic split and create a test set**
   - Current split is random and done inside `convert_to_yolo.py`.
   - Add a fixed seed and create `train/val/test` splits.

3. **Increase dataset diversity**
   - Add images with different:
     - lighting/time-of-day
     - camera distances
     - occlusions
     - backgrounds
     - defect types and severities

4. **Try a larger model if compute allows**
   - Swap to `yolov8s.pt` or `yolov8m.pt`.
   - On CPU this may be slow; on GPU it is typically practical.

5. **Tune augmentations and training schedule**
   - `train2` uses default degrees=0 and flipud=0.
   - `train3` attempted stronger augmentation (degrees=90, flipud=0.5) but appears incomplete.
   - Consider moderate augmentations aligned to your camera geometry.

6. **Error analysis using saved artifacts**
   - Use:
     - `confusion_matrix_normalized.png`
     - `val_batch*_pred.jpg`
   - Identify whether errors are:
     - class confusion (normal vs defective)
     - localization errors (boxes off)
     - missed detections (recall problem)

7. **Adjust inference thresholds for your use case**
   - For demo visualization, low `conf` is fine.
   - For production, increase `conf` and tune NMS IoU to balance false positives vs false negatives.

## Project “story” for a presentation
If you want a clean narrative:

1. Problem: detect and classify insulators as normal/defective.
2. Data: CPLID-style images + VOC annotations.
3. Conversion: VOC XML -> YOLO txt + train/val split.
4. Model: YOLOv8n transfer learning.
5. Training: 50 epochs @ 640px.
6. Results: mAP50 ~ 0.98, mAP50-95 ~ 0.86 on the validation split.
7. Next steps: stronger evaluation split, more data diversity, bigger model, and systematic tuning.
