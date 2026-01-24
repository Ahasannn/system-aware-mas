#!/bin/bash
python Experiments/run_mbpp.py \
  --test_limit 3 \
  --concurrency 2 \
  --arrival-rate 1 2 3\
  --checkpoint checkpoints/mas_router/mas_mbpp_train_6.pth \
  --epochs 0
