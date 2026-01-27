#!/bin/bash
# System Router Testing Script for MBPP Dataset

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/system_router"

# Dataset directory (offline)
export MBPP_DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# ============================================================================
# Load Configuration: [arrival_pattern, concurrency, arrival_rate]
# ============================================================================
CONFIGS=(
  "poisson,5,5"
  "poisson,20,20"
  "microburst,30,30"
)

# Use the trained checkpoint (or override with argument)
CHECKPOINT_PATH="${1:-${CHECKPOINT_DIR}/system_router_mbpp.pt}"
TELEMETRY_CSV="logs/system_router_mbpp_test.csv"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
  echo "Please train the model first or provide a checkpoint path."
  echo "Usage: $0 [checkpoint_path]"
  exit 1
fi

echo "Using checkpoint: ${CHECKPOINT_PATH}"

# Delete existing telemetry to start fresh
if [ -f "${TELEMETRY_CSV}" ]; then
  echo "Removing existing telemetry: ${TELEMETRY_CSV}"
  rm -f "${TELEMETRY_CSV}"
fi

# Loop through configurations
for config in "${CONFIGS[@]}"; do
  IFS=',' read -r PATTERN CONCURRENCY RATE <<< "${config}"

  echo "=============================================="
  echo "Testing with: pattern=${PATTERN} concurrency=${CONCURRENCY} rate=${RATE}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Telemetry:  ${TELEMETRY_CSV}"
  echo "=============================================="

  python Experiments/train_system_router_mbpp.py \
    --split test \
    --limit 200 \
    --epochs 1 \
    --max-tokens 512 \
    --seed 42 \
    --deterministic \
    --arrival-pattern "${PATTERN}" \
    --concurrency "${CONCURRENCY}" \
    --arrival-rate "${RATE}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --resume-checkpoint \
    --telemetry-csv "${TELEMETRY_CSV}"

  echo ""
  echo "Completed: pattern=${PATTERN} concurrency=${CONCURRENCY} rate=${RATE}"
  echo ""
done

echo "All test configurations completed!"
