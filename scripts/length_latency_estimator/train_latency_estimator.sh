#!/bin/bash
# Train the latency estimator model
# This model predicts TTFT (time to first token) and TPOT (time per output token)
# based on system metrics (queue depth, KV cache usage, etc.)

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint directory
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/estimators"
mkdir -p "${CHECKPOINT_DIR}"

# ============================================================
# CONFIGURE CSV PATH HERE
# ============================================================
# Path to the training CSV file (from system router telemetry)
# Expected columns: prompt_tokens, llm_waiting, llm_running, llm_kv_cache_usage,
#                   llm_itl_avg, llm_ttft_avg, observed_ttft, observed_tpot,
#                   model_name, role_name, strategy_name
CSV_PATH="/home/ji757406.ucf/ahasan/system-aware-mas/logs/temp_test/system_router_mbpp_test.csv"  # <-- Set your CSV path here

# ============================================================
# Training hyperparameters
# ============================================================
BATCH_SIZE=32
EPOCHS=10
LEARNING_RATE=1e-3
MIN_TTFT=1e-6
MIN_TPOT=1e-6
RECORD_TYPE="role_step"

# Output checkpoint path
OUTPUT_PATH="${CHECKPOINT_DIR}/latency_estimator.pt"

# ============================================================
# Validation
# ============================================================
if [ -z "${CSV_PATH}" ]; then
    echo "ERROR: CSV_PATH is not set. Please set the CSV_PATH variable in this script."
    exit 1
fi

if [ ! -f "${CSV_PATH}" ]; then
    echo "ERROR: CSV file not found: ${CSV_PATH}"
    exit 1
fi

# ============================================================
# Delete existing checkpoint if present
# ============================================================
if [ -f "${OUTPUT_PATH}" ]; then
    echo "Removing existing checkpoint: ${OUTPUT_PATH}"
    rm -f "${OUTPUT_PATH}"
fi

# ============================================================
# Run training
# ============================================================
echo "Training Latency Estimator"
echo "=========================="
echo "CSV Path: ${CSV_PATH}"
echo "Output Path: ${OUTPUT_PATH}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo ""

python Experiments/train_latency_estimator.py \
    --csv-path "${CSV_PATH}" \
    --record-type "${RECORD_TYPE}" \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --min-ttft ${MIN_TTFT} \
    --min-tpot ${MIN_TPOT} \
    --output-path "${OUTPUT_PATH}"

echo ""
echo "Training complete. Checkpoint saved to: ${OUTPUT_PATH}"
