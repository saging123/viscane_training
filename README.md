# Sugarcane Variety Classifier CLI

This project now includes a complete CLI pipeline to:
1. validate/clean your folder-by-variety dataset,
2. split it into `train/val/test`,
3. train a classifier (ResNet18 transfer learning),
4. save model + metrics.

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
  --seed 42
```

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
  --resize 256
```

### Train only (after preprocess)

```bash
python main.py train \
  --prepared-dir data/prepared \
  --output-dir artifacts \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.001 \
  --image-size 224 \
  --workers 4 \
  --seed 42
```

### Test only (evaluate saved model on test split)

```bash
python main.py test \
  --prepared-dir data/prepared \
  --checkpoint-path artifacts/best_model.pt \
  --batch-size 32 \
  --workers 4
```

### End-to-end (preprocess + train)

```bash
python main.py all \
  --raw-dir /path/to/your_raw_dataset \
  --prepared-dir data/prepared \
  --output-dir artifacts \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --label-mode variety_maturity \
  --resize 256 \
  --epochs 25 \
  --batch-size 32 \
  --lr 0.001 \
  --image-size 224 \
  --workers 4 \
  --seed 42
```

## Output artifacts

- `artifacts/best_model.pt`: best checkpoint (by validation accuracy)
- `artifacts/metrics.json`: classes and training metrics

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
    epochs=25,
    batch_size=32,
    image_size=224,
    workers=2,
)

print(prep)
print(train)
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

set `--label-mode variety_maturity` (or `label_mode="variety_maturity"` in Colab).
The model trains on combined labels (example: `variety_1__mature`) and metrics include decoded fields for both `variety` and `maturity_status`.
