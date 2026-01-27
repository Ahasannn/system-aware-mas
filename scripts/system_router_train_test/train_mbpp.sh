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
  "sustained,30,30"
  "microburst,50,50"
  "sustained,100,100"
  "microburst,200,200"
)

# Single checkpoint and telemetry file (improved across all configs)
CHECKPOINT_PATH="${CHECKPOINT_DIR}/system_router_mbpp.pt"
TELEMETRY_CSV="logs/temp_test/system_router_mbpp.csv"

# Delete existing checkpoint to start fresh (avoids architecture mismatch errors)
if [ -f "${CHECKPOINT_PATH}" ]; then
  echo "Removing existing checkpoint: ${CHECKPOINT_PATH}"
  rm -f "${CHECKPOINT_PATH}"
fi

# Delete existing telemetry to start fresh
if [ -f "${TELEMETRY_CSV}" ]; then
  echo "Removing existing telemetry: ${TELEMETRY_CSV}"
  rm -f "${TELEMETRY_CSV}"
fi

# Loop through configurations
FIRST_CONFIG=true
for config in "${CONFIGS[@]}"; do
  IFS=',' read -r PATTERN CONCURRENCY RATE <<< "${config}"

  echo "=============================================="
  echo "Training with: pattern=${PATTERN} concurrency=${CONCURRENCY} rate=${RATE}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Telemetry:  ${TELEMETRY_CSV}"
  echo "=============================================="

  # Build command - resume from checkpoint after first config
  CMD="python Experiments/train_system_router_mbpp.py \
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
    --telemetry-csv ${TELEMETRY_CSV}"

  if [ "${FIRST_CONFIG}" = false ]; then
    CMD="${CMD} --resume-checkpoint"
  fi

  eval "${CMD}"
  FIRST_CONFIG=false

  echo ""
  echo "Completed: pattern=${PATTERN} concurrency=${CONCURRENCY} rate=${RATE}"
  echo ""
done

echo "All configurations completed!"
