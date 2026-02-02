#!/bin/bash
# Baseline MAS Test Arrival Rate Sweep for MATH
# This script loops through arrival rate and concurrency pairs and saves results to a single CSV

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Dataset directory (offline)
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Output CSV file
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_math_test_temp.csv"

# Arrival patterns, rates, and concurrency tuples: (pattern, arrival_rate, concurrency)
# Using same rates as MBPP for consistency in motivation plots
RUN_CONFIGS=(
    "sustained 2 1000"
    "sustained 20 1000"
)

# Loop through each configuration
for pair in "${RUN_CONFIGS[@]}"; do
    read -r pattern arrival_rate concurrency <<< "$pair"

    echo "========================================"
    echo "Running with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo "========================================"

    python Experiments/run_math.py \
        --concurrency "$concurrency" \
        --arrival-rate "$arrival_rate" \
        --arrival-pattern "$pattern" \
        --checkpoint "${CHECKPOINT_DIR}/mas_math_train_full.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0 \
        --batch_size 1 \
        --test_limit 1000

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
