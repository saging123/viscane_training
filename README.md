# Sugarcane Variety Classifier CLI

This project now includes a complete CLI pipeline to:
1. validate/clean your folder-by-variety dataset,
2. split it into `train/val/test`,
3. train a classifier with either ResNet18 or YOLOv8,
4. save model, metrics, and Android-friendly exports.

## 1) Dataset structure (raw)

Put images like this:

```text
your_raw_dataset/
  variety_A/
    img1.jpg
    img2.jpg
  variety_B/
    img1.jpg
    ...
```

## 2) Install packages

```bash
pip install -r requirements.txt
```

## 3) CLI commands

Use `main.py` as the entrypoint.

### Preprocess only

```bash
python main.py preprocess \
  --raw-dir /path/to/your_raw_dataset \
  --prepared-dir data/prepared \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --label-mode variety_maturity \
  --resize 256 \
  --preprocess-device auto \
  --preprocess-workers 4 \
  --seed 42
```

The split logic is group-aware, so obvious filename variants and exact duplicate
captures stay in the same split instead of leaking across `train/val/test`.

`--preprocess-device auto` uses CUDA for resize preprocessing when a GPU is available.
Use `--preprocess-device cuda` to require GPU, or `--preprocess-device cpu` to force CPU.
Use `--preprocess-device cpu --preprocess-workers 4` or higher to crop/resize images
with multiple CPU worker processes.

### Preprocess to folder-per-variety output (no split)

If your goal is:
- input: `raw/<variety_name>/images...`
- output: `processed/<variety_name>/images...`

use:

```bash
python main.py preprocess-flat \
  --raw-dir raw \
  --processed-dir processed \
  --label-mode variety_maturity \
  --resize 256 \
  --preprocess-device auto \
  --preprocess-workers 4
```

### Train only (after preprocess)

```bash
python main.py train \
  --prepared-dir data/prepared \
  --output-dir artifacts/resnet18 \
  --model-type resnet18 \
  --epochs 35 \
  --batch-size 32 \
  --lr 0.0005 \
  --weight-decay 0.0005 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.002 \
  --image-size 224 \
  --workers 4 \
  --seed 42
```

Train YOLOv8 classification on the same prepared split:

```bash
python main.py train \
  --prepared-dir data/prepared \
  --output-dir artifacts/yolov8 \
  --model-type yolov8 \
  --yolo-weights yolov8n-cls.pt \
  --epochs 35 \
  --batch-size 32 \
  --lr 0.0005 \
  --weight-decay 0.0005 \
  --early-stopping-patience 8 \
  --image-size 224 \
  --workers 4 \
  --seed 42
```

During training and testing, resize/crop/flip/color jitter/normalization run on the
same device as the model. With CUDA available, the repeated per-epoch preprocessing
uses the GPU instead of your CPU.

For tougher research settings, the training pipeline already enables stronger
augmentation defaults for ResNet18 and YOLOv8 so performance is less inflated by
easy samples.

The training pipeline also now enables:
- early stopping, so runs stop when validation accuracy stops improving,
- inverse-frequency class weighting for ResNet18 by default, so weak classes have more influence on the loss,
- metrics logging for these controls in the saved `metrics.json` files.

### Audit split leakage

```bash
python main.py audit-splits \
  --prepared-dir data/prepared \
  --near-duplicate-distance 5 \
  --max-examples 25 \
  --workers 4
```

This writes `data/prepared/split_leakage_audit.json` so you can cite exact-duplicate
and near-duplicate leakage findings in your technical documentation.

### Analyze prepared dataset balance

```bash
python main.py analyze-prepared \
  --prepared-dir data/prepared \
  --low-sample-threshold 20
```

This writes `data/prepared/prepared_dataset_analysis.json` with:
- per-split image counts,
- per-class totals,
- minority-to-majority class ratio,
- low-sample warnings for weak classes,
- variety and maturity rollups when using `variety_maturity`.

Available label modes:
- `variety`: groups all maturity folders under each variety into one class
- `maturity`: groups the same maturity stage across all varieties into one class
- `variety_maturity`: keeps combined labels like `524__MATURE`

### Test only (evaluate saved model on test split)

```bash
python main.py test \
  --prepared-dir data/prepared \
  --checkpoint-path artifacts/best_model.pt \
  --batch-size 32 \
  --workers 4
```

For YOLOv8 checkpoints, pass the model type:

```bash
python main.py test \
  --prepared-dir data/prepared \
  --checkpoint-path artifacts/yolov8/yolov8/weights/best.pt \
  --model-type yolov8 \
  --batch-size 32 \
  --workers 4
```

### End-to-end (preprocess + train)

```bash
python main.py all \
  --raw-dir /path/to/your_raw_dataset \
  --prepared-dir data/prepared \
  --output-dir artifacts/resnet18 \
  --model-type resnet18 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --label-mode variety_maturity \
  --resize 256 \
  --preprocess-device auto \
  --preprocess-workers 4 \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.001 \
  --image-size 224 \
  --workers 4 \
  --seed 42
```

If `data/prepared` is already built and you only want to adjust training settings,
skip preprocessing:

```bash
python main.py all \
  --skip-preprocess \
  --prepared-dir data/prepared \
  --output-dir artifacts \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.001 \
  --image-size 224 \
  --workers 4 \
  --seed 42
```

## Output artifacts

- `artifacts/resnet18/best_model.pt`: ResNet18 checkpoint (by validation accuracy)
- `artifacts/resnet18/resnet18_android.ptl`: PyTorch Lite artifact for Android
- `artifacts/resnet18/resnet18_android.onnx`: ONNX export for Android runtimes
- `artifacts/resnet18/resnet18_android_metadata.json`: Android preprocessing and labels
- `artifacts/yolov8/yolov8/weights/best.pt`: YOLOv8 classification checkpoint
- `artifacts/yolov8/yolov8/weights/best.onnx`: YOLOv8 ONNX export when export succeeds
- `artifacts/yolov8/yolov8_android_metadata.json`: Android preprocessing and labels
- `artifacts/metrics.json`: classes and training metrics

## Serve the Trained Model as a FastAPI API

After training, start the API from the project root:

```bash
uvicorn sugarcane_variety.api:app --host 0.0.0.0 --port 8000
```

To keep the API running in the background after you close the terminal:

```bash
nohup uvicorn sugarcane_variety.api:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

Check the server log:

```bash
tail -f uvicorn.log
```

Stop the background server:

```bash
pkill -f "uvicorn sugarcane_variety.api:app"
```

By default, the API loads:

```text
artifacts/best_model.pt
```

To use a different checkpoint:

```bash
MODEL_CHECKPOINT=/path/to/best_model.pt \
uvicorn sugarcane_variety.api:app --host 0.0.0.0 --port 8000
```

Open the interactive docs:

```text
http://localhost:8000/docs
```

Useful endpoints:

- `GET /health`: check whether the model loaded.
- `GET /models`: list supported model pipelines and Android export formats.
- `POST /models/load`: load a specific ResNet18 or YOLOv8 checkpoint.
- `GET /reports/current`: get live training progress and saved research reports.
- `GET /classes`: list class labels.
- `POST /predict`: upload one image and get the predicted sugarcane class.
- `POST /training/start`: start preprocessing/training in the background.
- `GET /training/status`: check the current training job status.
- `GET /artifacts/download`: download `.pt` and `.json` artifact files as a ZIP.
- `GET /reports/model-comparison`: compare ResNet18 and YOLOv8 metrics/curves in one page.
- `GET /reports/technical-documentation`: open the research-oriented technical report page.

Start training through the API:

```bash
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_dir": "content/data/raw",
    "prepared_dir": "content/data/prepared",
    "output_dir": "content/data/sugarcane_artifacts",
    "model_type": "resnet18",
    "label_mode": "variety_maturity",
    "epochs": 25,
    "batch_size": 32,
    "perform_preprocess": true
}'
```

Start YOLOv8 training through the API while reusing an existing prepared split:

```bash
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "prepared_dir": "content/data/prepared",
    "output_dir": "content/data/sugarcane_artifacts/yolov8",
    "model_type": "yolov8",
    "yolo_weights": "yolov8n-cls.pt",
    "epochs": 25,
    "batch_size": 32,
    "perform_preprocess": false
  }'
```

Load a specific checkpoint for prediction:

```bash
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "content/data/sugarcane_artifacts/yolov8/yolov8/weights/best.pt",
    "model_type": "yolov8"
  }'
```

Check training status:

```bash
curl "http://localhost:8000/training/status"
```

Get the current research/comparison report while training is running or after it
finishes:

```bash
curl "http://localhost:8000/reports/current"
```

To read reports from a specific artifact folder:

```bash
curl "http://localhost:8000/reports/current?artifacts_dir=content/data/sugarcane_artifacts/yolov8"
```

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -F "file=@/path/to/sugarcane_leaf.jpg"
```

Example response:

```json
{
  "filename": "sugarcane_leaf.jpg",
  "prediction": {
    "class_index": 0,
    "confidence": 0.94,
    "class_name": "variety_1__mature",
    "variety": "variety_1",
    "maturity_status": "mature"
  },
  "top_k": [

  ## Download a Google Drive dataset folder

  If your dataset is shared from Google Drive as a folder, use the root script:

  ```bash
  python gdrive_downloader.py \
    <DRIVE_FOLDER_ID_OR_URL> \
    --output-dir /content/data/raw
  ```

  The script uses `gdown` to download a folder into the local dataset directory.
  Install dependencies first:

  ```bash
  pip install -r requirements.txt
  ```

    {
      "class_index": 0,
      "confidence": 0.94,
      "class_name": "variety_1__mature",
      "variety": "variety_1",
      "maturity_status": "mature"
    }
  ]
}
```

## Google Colab Compatible Usage

Use the helper file:
- `sugarcane_variety/colab_compatible.py`

Example Colab cells:

```python
import os

REPO_URL = "<your-repo-url>"
REPO_DIR = "<your-repo-folder>"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}

%cd {REPO_DIR}
!git pull
```

```python
from sugarcane_variety.colab_compatible import (
    install_requirements,
    mount_drive,
    print_eval_summary,
    run_all_for_colab,
    test_for_colab,
)

install_requirements("requirements.txt")
mount_drive("/content/drive")
```

```python
prep, train = run_all_for_colab(
    raw_dir="/content/drive/MyDrive/sugarcane_raw",   # folder-by-variety
    prepared_dir="/content/data/prepared",
    output_dir="/content/drive/MyDrive/sugarcane_artifacts",
    label_mode="variety_maturity",
    preprocess_device="auto",  # uses Colab GPU when available
    epochs=25,
    batch_size=32,
    image_size=224,
    workers=2,
    model_type="resnet18",
)

print(prep)
print(train)
```

To compare with YOLOv8 without repeating preprocessing:

```python
_, yolo_train = run_all_for_colab(
    raw_dir="/content/drive/MyDrive/sugarcane_raw",
    prepared_dir="/content/data/prepared",
    output_dir="/content/drive/MyDrive/sugarcane_artifacts/yolov8",
    label_mode="variety_maturity",
    perform_preprocess=False,
    model_type="yolov8",
    yolo_weights="yolov8n-cls.pt",
    epochs=25,
    batch_size=32,
    image_size=224,
    workers=2,
)
```

Test again later in Colab (without retraining):

```python
eval_result = test_for_colab(
    prepared_dir="/content/data/prepared",
    checkpoint_path="/content/drive/MyDrive/sugarcane_artifacts/best_model.pt",
    batch_size=32,
    workers=2,
)

print(eval_result)
print_eval_summary(eval_result)
```

`test` now gives a detailed, interpretation-friendly summary:
- exact label accuracy (`variety + maturity`)
- variety-only accuracy
- maturity-only accuracy (if available)
- per-class performance in `test_summary.json`
- top confusion pairs
- friendly outcome message

For your dataset structure:

```text
raw/
  variety_1/
    mature/
    not_mature/
  variety_2/
    mature/
    not_mature/
```

set:
- `--label-mode variety` for variety-only experiments,
- `--label-mode maturity` for maturity-only experiments,
- `--label-mode variety_maturity` for combined-label experiments.

The model trains on:
- `variety` labels like `524` or `847` for variety-only experiments,
- `maturity` labels like `MATURE` or `NOT_MATURE` for maturity-only experiments,
- combined labels like `524__MATURE` for joint experiments.
