#!/bin/bash
# Baseline MAS training for MATH

# Repo root (script is in scripts/baseline_train)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# Checkpoint and CSV paths
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_full.pth"
TELEMETRY_DIR="logs/baseline_mas_training/math"
TELEMETRY_CSV="${TELEMETRY_DIR}/mas_train_math_full_last.csv"

# Create logs directory if needed
mkdir -p logs "${TELEMETRY_DIR}"

# Dataset root (override with MATH_DATASET_ROOT)
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "MATH dataset not found at ${MATH_DATASET_ROOT}."
    echo "Expected ${MATH_DATASET_ROOT}/train and ${MATH_DATASET_ROOT}/test."
    echo "Download and extract the dataset there, or set MATH_DATASET_ROOT."
    exit 1
fi

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# Cost-quality tradeoff coefficient (override with COST_RATE env var)
COST_RATE="${COST_RATE:-700.0}"

# Build command with conditional checkpoint loading
CMD="python Experiments/run_math.py \
  --epochs 1 \
  --batch_size 8 \
  --lr 0.01 \
  --test_limit 16 \
  --cost_rate ${COST_RATE} \
  --dataset-root ${MATH_DATASET_ROOT} \
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
