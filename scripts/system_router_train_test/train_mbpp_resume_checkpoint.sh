#!/bin/bash
# System Router Training Script for MBPP Dataset

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/system_router_full"
mkdir -p "${CHECKPOINT_DIR}"

# Dataset directory (offline)
export MBPP_DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

# ============================================================================
# Load Configuration: [arrival_pattern, concurrency, arrival_rate]
# ============================================================================
CONFIGS=(
  "poisson,15,15"
  "poisson,40,40"
  "poisson,2,2"
  "poisson,60,60"
)

# Single checkpoint and telemetry file (improved across all configs)
CHECKPOINT_PATH="${CHECKPOINT_DIR}/system_router_mbpp.pt"
TELEMETRY_CSV="logs/temp_test/system_router_mbpp.csv"

# Resume from existing checkpoint and append to telemetry CSV
if [ -f "${CHECKPOINT_PATH}" ]; then
  echo "Resuming from existing checkpoint: ${CHECKPOINT_PATH}"
else
  echo "No existing checkpoint found, starting fresh"
fi

if [ -f "${TELEMETRY_CSV}" ]; then
  echo "Appending to existing telemetry: ${TELEMETRY_CSV}"
fi

# Loop through configurations
for config in "${CONFIGS[@]}"; do
  IFS=',' read -r PATTERN CONCURRENCY RATE <<< "${config}"

  echo "=============================================="
  echo "Training with: pattern=${PATTERN} concurrency=${CONCURRENCY} rate=${RATE}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Telemetry:  ${TELEMETRY_CSV}"
  echo "=============================================="

  # Build command - always resume from checkpoint if it exists
  python Experiments/train_system_router_mbpp.py \
    --split train \
    --epochs 1 \
    --max-tokens 512 \
    --limit 0 \
    --seed 42 \
    --arrival-pattern ${PATTERN} \
    --concurrency ${CONCURRENCY} \
    --arrival-rate ${RATE} \
    --checkpoint-path ${CHECKPOINT_PATH} \
    --checkpoint-every 50 \
    --telemetry-csv ${TELEMETRY_CSV} \
    --resume-checkpoint

  echo ""
  echo "Completed: pattern=${PATTERN} concurrency=${CONCURRENCY} rate=${RATE}"
  echo ""
done

echo "All configurations completed!"
