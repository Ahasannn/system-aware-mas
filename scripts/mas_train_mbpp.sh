#!/bin/bash
python Experiments/run_mbpp.py \
  --epochs 1 \
  --batch_size 3 \
  --lr 0.01 \
  --train_limit 6 \
  --test_limit 1 \
  --train-telemetry-csv logs/my_train_output_6.csv \
  --save-checkpoint checkpoints/mas_router/mas_mbpp_train_6.pth
