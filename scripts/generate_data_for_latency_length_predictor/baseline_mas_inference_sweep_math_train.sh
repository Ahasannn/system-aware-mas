#!/bin/bash
# Baseline MAS Inference Sweep on MATH Training Data
# Purpose: Collect latency/metrics data for latency-length predictor training.
# Runs inference-only (epochs=0) with a pre-trained checkpoint on training data
# under varying arrival rates to observe latency under different loads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Blue storage configuration
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# Dataset directory (offline)
export MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# ===== Load shared dataset config =====
DATASET_CONFIG="${REPO_ROOT}/Experiments/dataset_config.json"
DATASET_NAME="math"

TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['train_limit'])")
ARRIVAL_PATTERN=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_pattern'])")
ARRIVAL_RATES=$(python3 -c "import json; print(' '.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_rates']))")

# Concurrency (max simultaneous threads — high value lets vLLM scheduler manage)
CONCURRENCY=1000

# Build run configs from shared dataset config
RUN_CONFIGS=()
for rate in $ARRIVAL_RATES; do
    RUN_CONFIGS+=("${ARRIVAL_PATTERN} ${rate} ${CONCURRENCY}")
done

# Output directory and CSV file
OUTPUT_DIR="logs/generate_data_for_latency_length_predictor"
mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="${OUTPUT_DIR}/baseline_mas_inference_math_train_${TRAIN_LIMIT}_${ARRIVAL_PATTERN}.csv"

echo "========================================"
echo "Latency-Length Predictor Data Generation"
echo "Dataset: MATH (train split, ${TRAIN_LIMIT} stratified samples)"
echo "Rates:   ${ARRIVAL_RATES}"
echo "Pattern: ${ARRIVAL_PATTERN}"
echo "Output:  ${OUTPUT_CSV}"
echo "========================================"
echo ""

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
        --test-split train \
        --generate-predictor-data \
        --epochs 0 \
        --test_limit "$TRAIN_LIMIT"

    echo "Completed run with arrival_rate=$arrival_rate, concurrency=$concurrency, pattern=$pattern"
    echo ""
done

echo "========================================"
echo "All sweeps completed!"
echo "Results saved to: $OUTPUT_CSV"
echo "========================================"
