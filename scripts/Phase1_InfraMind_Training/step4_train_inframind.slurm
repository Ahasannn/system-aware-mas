#!/bin/bash
#SBATCH --job-name=inframind_step4_train
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step4_train/slurm-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step4_train/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# =========================================================================
# Step 4: Phase 2 InfraMind training — uses all 3 predictors, fresh from MAS
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
PREDICTOR_DIR="${PREDICTOR_DIR:-${BLUE_STORAGE}/checkpoints/predictors}"
MAS_CHECKPOINT="${MAS_CHECKPOINT:-${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth}"
LATENCY_PREDICTOR="${PREDICTOR_DIR}/latency_estimator.pth"
LENGTH_PREDICTOR="${PREDICTOR_DIR}/length_estimator.pth"
QUALITY_PREDICTOR="${PREDICTOR_DIR}/quality_estimator.pth"

LIMIT="${LIMIT:-500}"
EPOCHS="${EPOCHS:-3}"
ARRIVAL_RATES="${ARRIVAL_RATES:-0,10,50,100,200,500}"
BUDGET_SWEEP="${BUDGET_SWEEP:-20,60,150}"
CONCURRENCY="${CONCURRENCY:-4}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/inframind_${DATASET}_phase2.pt"

LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step4_train"
mkdir -p "${STEP_DIR}"
TELEMETRY_CSV="${STEP_DIR}/train_${SLURM_JOB_ID}.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# --- Verify predictors exist ---
echo "Checking predictor checkpoints..."
for ckpt in "${LATENCY_PREDICTOR}" "${LENGTH_PREDICTOR}" "${QUALITY_PREDICTOR}"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Missing predictor checkpoint: $ckpt"
        echo "Run step3_train_predictors.slurm first."
        exit 1
    fi
    echo "  Found: $ckpt ($(du -h "$ckpt" | cut -f1))"
done
echo ""

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

# --- Phase 2 training ---
echo "========================================="
echo "Phase 2 InfraMind Training"
echo "Dataset: ${DATASET}"
echo "MAS checkpoint: ${MAS_CHECKPOINT}"
echo "Quality predictor: ${QUALITY_PREDICTOR}"
echo "Latency predictor: ${LATENCY_PREDICTOR}"
echo "Length predictor: ${LENGTH_PREDICTOR}"
echo "========================================="

CMD="python Experiments/train_inframind_${DATASET}.py \
  --epochs ${EPOCHS} \
  --limit ${LIMIT} \
  --arrival-rates ${ARRIVAL_RATES} \
  --budget-sweep ${BUDGET_SWEEP} \
  --concurrency ${CONCURRENCY} \
  --training-batch-size ${TRAINING_BATCH_SIZE} \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --quality-predictor ${QUALITY_PREDICTOR} \
  --checkpoint-path ${CHECKPOINT_PATH} \
  --telemetry-csv ${TELEMETRY_CSV} \
  --dataset-root ${DATASET_ROOT}"

echo "Command: $CMD"
set +e
$CMD
EXIT_CODE=$?
set -e

# Create well-known symlink for easy access
LATEST_LINK="${STEP_DIR}/train.csv"
if [ $EXIT_CODE -eq 0 ] && [ -f "${TELEMETRY_CSV}" ]; then
    ln -sf "$(basename "${TELEMETRY_CSV}")" "${LATEST_LINK}"
    echo "Symlinked: ${LATEST_LINK} -> $(basename "${TELEMETRY_CSV}")"
fi

echo ""
echo "========================================="
echo "Step 4 completed with exit code: $EXIT_CODE"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Telemetry: ${TELEMETRY_CSV}"
echo "Latest link: ${LATEST_LINK}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
