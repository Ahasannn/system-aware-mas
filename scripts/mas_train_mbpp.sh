#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# Checkpoint and CSV paths
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_mbpp_train_full.pth"
TELEMETRY_CSV="logs/baseline_mas_training/mas_train_mbpp_telemetry.csv"

# Create logs directory if needed
mkdir -p logs

# Dataset directory (offline)
export MBPP_DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# Build command with conditional checkpoint loading
CMD="python Experiments/run_mbpp.py \
  --epochs 1 \
  --batch_size 32 \
  --lr 0.01 \
  --test_limit 1 \
  --train-telemetry-csv ${TELEMETRY_CSV} \
  --save-checkpoint ${CHECKPOINT_PATH}"

# If checkpoint exists, resume from it
if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Found existing checkpoint: ${CHECKPOINT_PATH}"
    echo "Resuming training..."
    CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
else
    echo "No existing checkpoint found. Starting fresh training..."
fi

# Run training
echo "Running: ${CMD}"
eval ${CMD}
