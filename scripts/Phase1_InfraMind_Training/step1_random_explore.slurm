#!/bin/bash
#SBATCH --job-name=inframind_step1_explore
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step1_explore/slurm-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step1_explore/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# =========================================================================
# Step 1: Random Exploration — generate diverse (model, strategy) CSV data
#         Planner uses learned MAS weights for topology/role selection.
#         Executor uses uniform-random model + strategy selection.
# =========================================================================

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# --- Configuration ---
DATASET="${DATASET:-math}"
DATASET_ROOT="${DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
MAS_CHECKPOINT="${MAS_CHECKPOINT:-${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth}"
LIMIT="${LIMIT:-500}"
ARRIVAL_RATES="${ARRIVAL_RATES:-100,200}"
BUDGET_SWEEP="${BUDGET_SWEEP:-60}"
CONCURRENCY="${CONCURRENCY:-1000}"
EPOCHS="${EPOCHS:-1}"

LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step1_explore"
mkdir -p "${STEP_DIR}"
TELEMETRY_CSV="${STEP_DIR}/explore_${SLURM_JOB_ID}.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# --- vLLM cleanup ---
cleanup_vllm() {
    echo ""
    echo "Cleaning up vLLM servers..."
    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"
    if ls "${vllm_log_dir}"/*.pid >/dev/null 2>&1; then
        for pidfile in "${vllm_log_dir}"/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                name=$(basename "$pidfile" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Stopping $name (PID $pid)"
                    kill -TERM "$pid" 2>/dev/null
                    sleep 2
                    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
                fi
                rm -f "$pidfile"
            fi
        done
    fi
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

# --- Start vLLM ---
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

# --- Run random exploration ---
echo "========================================="
echo "Running random exploration..."
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
