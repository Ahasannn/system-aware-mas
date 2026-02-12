#!/bin/bash
# train_inframind_math.sh
# Train the InfraMind (CMDP) on the MATH dataset.
# Expects vLLM servers already running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source .venv/bin/activate

# ===== Load shared dataset config =====
DATASET_CONFIG="${REPO_ROOT}/Experiments/dataset_config.json"
DATASET_NAME="math"

TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['train_limit'])")
ARRIVAL_PATTERN=$(python3 -c "import json; print(json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_pattern'])")
ARRIVAL_RATES_CSV=$(python3 -c "import json; print(','.join(str(r) for r in json.load(open('${DATASET_CONFIG}'))['${DATASET_NAME}']['arrival_rates']))")

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"

# Pretrained MAS Router checkpoint (transfer learning for planner)
MAS_CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth"

# Predictor checkpoints
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"

# Budget sweep values (seconds) — sweep through fixed budgets instead of per-query CSV
# 3 budgets: tight, medium, generous
BUDGET_SWEEP="20,60,150"

# Override arrival rates for training (3 rates: low, medium, high)
# Full config has 6 rates which is too slow; can expand after validation
ARRIVAL_RATES_CSV="10,100,200"

# Dataset
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

# Checkpoint directory (filename auto-includes run_id with SLURM_JOB_ID)
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"

# To resume from a previous run, set this to the checkpoint path:
RESUME_CHECKPOINT="/blue/qi855292.ucf/ah872032.ucf/checkpoints/inframind/inframind_math_20260212_161547_job24816311.pt"

# ===== Validation =====
for f in "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "${MATH_DATASET_ROOT}/train" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

# ===== Environment =====
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

echo "========================================="
echo "InfraMind Training — MATH"
echo "========================================="
echo "Train limit: ${TRAIN_LIMIT}"
echo "Arrival rates: ${ARRIVAL_RATES_CSV}"
echo "Arrival pattern: ${ARRIVAL_PATTERN}"
echo "Budget sweep: ${BUDGET_SWEEP}"
echo "MAS checkpoint: ${MAS_CHECKPOINT}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "========================================="
echo ""

# ===== Training =====
CMD="python Experiments/train_inframind_math.py \
  --dataset-root ${MATH_DATASET_ROOT} \
  --limit ${TRAIN_LIMIT} \
  --epochs 3 \
  --max-tokens 4096 \
  --arrival-rates ${ARRIVAL_RATES_CSV} \
  --arrival-pattern ${ARRIVAL_PATTERN} \
  --budget-sweep ${BUDGET_SWEEP} \
  --concurrency 1000 \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --checkpoint-every 50"

# Load pretrained MAS planner weights (transfer learning)
if [[ -f "${MAS_CHECKPOINT}" ]]; then
    echo "Loading MAS planner weights: ${MAS_CHECKPOINT}"
    CMD="${CMD} --mas-checkpoint ${MAS_CHECKPOINT}"
fi

# Resume from a previous InfraMind checkpoint if specified
if [[ -n "${RESUME_CHECKPOINT}" && -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Resuming from checkpoint: ${RESUME_CHECKPOINT}"
    CMD="${CMD} --checkpoint-path ${RESUME_CHECKPOINT} --resume-checkpoint"
elif [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "WARNING: Resume checkpoint not found: ${RESUME_CHECKPOINT}"
fi

echo "Command: ${CMD}"
echo ""

$CMD
echo ""
echo "Training complete."
