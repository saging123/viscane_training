#!/bin/bash

# ===== CONFIG =====
VM_IP="35.221.191.84"
VM_USER="$USER"
LOCAL_DIR="/home/nissoftdev2/AI_AssistCode/viscane_train/"
REMOTE_DIR="~/"

# ===== SYNC =====
rsync -avz \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.env' \
  "$LOCAL_DIR" \
  "$VM_USER@$VM_IP:$REMOTE_DIR"

echo " Deploy complete"
