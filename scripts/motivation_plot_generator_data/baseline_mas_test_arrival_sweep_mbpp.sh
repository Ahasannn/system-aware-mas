#!/bin/bash
# Baseline MAS Test Arrival Rate Sweep for MBPP
# This script loops through arrival rate and concurrency pairs and saves results to a single CSV

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Dataset directory (offline)
export MBPP_DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

# Output CSV file
OUTPUT_CSV="logs/motivation_plot_generator_data/baseline_motivation_sweep_100.csv"

# Arrival rate and concurrency pairs: (arrival_rate, concurrency)
PAIRS=(
    "10 10"
    "30 30"
    "50 50"
    "70 70"
    "90 90"
    "120 120"
)

# Loop through each pair
for pair in "${PAIRS[@]}"; do
    read -r arrival_rate concurrency <<< "$pair"

    echo "========================================"
    echo "Running with arrival_rate=$arrival_rate, concurrency=$concurrency"
    echo "========================================"

    python Experiments/run_mbpp.py \
        --concurrency "$concurrency" \
        --arrival-rate "$arrival_rate" \
        --test_limit 100 \
        --checkpoint "${CHECKPOINT_DIR}/mas_mbpp_train_full_5ep.pth" \
        --test-telemetry-csv "$OUTPUT_CSV" \
        --epochs 0

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
