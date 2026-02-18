#!/bin/bash
# =========================================================================
# Step 1: Random Exploration (srun version — assumes vLLM already serving)
#         Planner uses learned MAS weights for topology/role selection.
#         Executor uses uniform-random model + strategy selection.
# =========================================================================

set -euo pipefail

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# --- Configuration ---
RUN_ID="${RUN_ID:-$(date +%s)}"
DATASET="${DATASET:-math}"
DATASET_ROOT="${DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
MAS_CHECKPOINT="${MAS_CHECKPOINT:-${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth}"
LIMIT="${LIMIT:-500}"
ARRIVAL_RATES="${ARRIVAL_RATES:-100,200,300,500,50,10,30,150}"
BUDGET_SWEEP="${BUDGET_SWEEP:-120}"
CONCURRENCY="${CONCURRENCY:-1000}"
EPOCHS="${EPOCHS:-1}"

LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step1_explore"
mkdir -p "${STEP_DIR}"
TELEMETRY_CSV="${STEP_DIR}/explore_${RUN_ID}.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# --- Verify vLLM is already running ---
echo "Checking vLLM status..."
bash scripts/check_vllm_status.sh || { echo "WARNING: Some vLLM servers may not be ready"; }
echo ""

# --- Run random exploration ---
echo "========================================="
echo "Running random exploration..."
echo "Run ID: ${RUN_ID}"
echo "Dataset: ${DATASET}"
echo "MAS checkpoint: ${MAS_CHECKPOINT}"
echo "Limit: ${LIMIT}"
echo "Arrival rates: ${ARRIVAL_RATES}"
echo "Budget sweep: ${BUDGET_SWEEP}"
echo "Output: ${TELEMETRY_CSV}"
echo "========================================="

CMD="python Experiments/train_inframind_${DATASET}.py \
  --random-exploration \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --epochs ${EPOCHS} \
  --limit ${LIMIT} \
  --arrival-rates ${ARRIVAL_RATES} \
  --budget-sweep ${BUDGET_SWEEP} \
  --concurrency ${CONCURRENCY} \
  --telemetry-csv ${TELEMETRY_CSV} \
  --dataset-root ${DATASET_ROOT}"

echo "Command: $CMD"
set +e
$CMD
EXIT_CODE=$?
set -e

# Create well-known symlink for step 2 to discover
LATEST_LINK="${STEP_DIR}/explore.csv"
if [ $EXIT_CODE -eq 0 ] && [ -f "${TELEMETRY_CSV}" ]; then
    ln -sf "$(basename "${TELEMETRY_CSV}")" "${LATEST_LINK}"
    echo "Symlinked: ${LATEST_LINK} -> $(basename "${TELEMETRY_CSV}")"
fi

echo ""
echo "========================================="
echo "Step 1 completed with exit code: $EXIT_CODE"
echo "Output CSV: ${TELEMETRY_CSV}"
echo "Latest link: ${LATEST_LINK}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
