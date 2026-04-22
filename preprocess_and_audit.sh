#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-content/data/raw/DATASETSFINAL}"
PREPARED_DIR="${2:-content/data/prepared}"
LABEL_MODE="${LABEL_MODE:-variety_maturity}"
RESIZE="${RESIZE:-256}"
PREPROCESS_DEVICE="${PREPROCESS_DEVICE:-auto}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-4}"
SEED="${SEED:-42}"
NEAR_DUPLICATE_DISTANCE="${NEAR_DUPLICATE_DISTANCE:-5}"
AUDIT_WORKERS="${AUDIT_WORKERS:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-25}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Running preprocess..."
echo "  raw_dir=$RAW_DIR"
echo "  prepared_dir=$PREPARED_DIR"
echo "  label_mode=$LABEL_MODE"
echo "  resize=$RESIZE"
echo "  preprocess_device=$PREPROCESS_DEVICE"
echo "  preprocess_workers=$PREPROCESS_WORKERS"
echo "  seed=$SEED"

"$PYTHON_BIN" main.py preprocess \
  --raw-dir "$RAW_DIR" \
  --prepared-dir "$PREPARED_DIR" \
  --label-mode "$LABEL_MODE" \
  --resize "$RESIZE" \
  --preprocess-device "$PREPROCESS_DEVICE" \
  --preprocess-workers "$PREPROCESS_WORKERS" \
  --seed "$SEED"

echo
echo "Running split audit..."
echo "  prepared_dir=$PREPARED_DIR"
echo "  near_duplicate_distance=$NEAR_DUPLICATE_DISTANCE"
echo "  workers=$AUDIT_WORKERS"
echo "  max_examples=$MAX_EXAMPLES"

"$PYTHON_BIN" main.py audit-splits \
  --prepared-dir "$PREPARED_DIR" \
  --near-duplicate-distance "$NEAR_DUPLICATE_DISTANCE" \
  --workers "$AUDIT_WORKERS" \
  --max-examples "$MAX_EXAMPLES"

echo
echo "Done. Check the audit JSON under:"
echo "  $PREPARED_DIR/split_leakage_audit.json"
