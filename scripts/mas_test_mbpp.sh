#!/bin/bash
# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Dataset directory (offline)
export MBPP_DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

python Experiments/run_mbpp.py \
  --test_limit 200 \
  --concurrency 150 \
  --arrival-rate 150 \
  --max_agent 5 \
  --checkpoint "${CHECKPOINT_DIR}/mas_mbpp_train_full_300.pth" \
  --test-telemetry-csv "logs/motivation_baseline_mbpp_arrival_rate_150.csv" \
  --epochs 0

