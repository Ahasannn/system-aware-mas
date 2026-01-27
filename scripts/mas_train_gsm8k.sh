#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"

# vLLM API configuration (EMPTY means no authentication required)
export KEY="EMPTY"

python Experiments/run_gsm8k.py \
  --epochs 1 \
  --batch_size 16 \
  --train_limit 300 \
  --lr 0.01 \
  --test_limit 1 \
  --train-telemetry-csv logs/gsm8k_train_output_full_300.csv \
  --save-checkpoint "${CHECKPOINT_DIR}/mas_gsm8k_train_full_300.pth"
