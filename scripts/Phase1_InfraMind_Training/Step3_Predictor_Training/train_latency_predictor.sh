#!/bin/bash
# =========================================================================
# Step 3c: Train Latency Predictor from judge-scored CSV
# Usage (inside srun session):
#   source .venv/bin/activate
#   bash scripts/Phase1_InfraMind_Training/Step3_Predictor_Training/train_latency_predictor.sh
# =========================================================================

set -euo pipefail

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh

echo "========================================="
echo "Step 3c: Train LATENCY Predictor"
echo "Start Time: $(date)"
echo "Python: $(which python)"
echo "========================================="

# --- Configuration ---
DATASET="${DATASET:-math}"
LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"

INPUT_CSV="${INPUT_CSV:-${LOG_BASE}/step2_judge/scored.csv}"
SAVE_DIR="${SAVE_DIR:-${BLUE_STORAGE}/checkpoints/inframind_predictors/${DATASET}}"
RECORD_TYPE="${RECORD_TYPE:-role_step}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"

mkdir -p "${SAVE_DIR}"

export TOKENIZERS_PARALLELISM="false"

echo "Input CSV:   ${INPUT_CSV}"
echo "Save dir:    ${SAVE_DIR}"
echo "Record type: ${RECORD_TYPE}"
echo "Epochs:      ${EPOCHS}"
echo "Batch size:  ${BATCH_SIZE}"
echo "LR:          ${LR}"
echo ""

CMD="python Experiments/train_latency_length_predictor.py \
  --csv ${INPUT_CSV} \
  --save-dir ${SAVE_DIR} \
  --record-type ${RECORD_TYPE} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --skip-length \
  --skip-quality"

echo "Command: $CMD"
echo "========================================="
$CMD
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Latency predictor training finished (exit code: $EXIT_CODE)"
echo "Checkpoint: ${SAVE_DIR}/latency_estimator.pth"
ls -la "${SAVE_DIR}"/latency_estimator.pth 2>/dev/null || echo "No checkpoint found"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
