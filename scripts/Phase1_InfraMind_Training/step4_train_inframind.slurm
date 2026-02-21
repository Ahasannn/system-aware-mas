#!/bin/bash
#SBATCH --job-name=inframind_step4_train
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step4_train/slurm-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step4_train/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# =========================================================================
# Step 4: InfraMind training — resume from best checkpoint
#   Phase 2b: budget-awareness focus with tighter budgets, lower max_num_seqs
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

LIMIT="${LIMIT:-519}"
VAL_LIMIT="${VAL_LIMIT:-131}"
EPOCHS="${EPOCHS:-40}"
ARRIVAL_RATES="${ARRIVAL_RATES:-10,50,100,200,400,150}"
BUDGET_MIN="${BUDGET_MIN:-5}"
BUDGET_MAX="${BUDGET_MAX:-120}"
CONCURRENCY="${CONCURRENCY:-1000}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
PATIENCE="${PATIENCE:-8}"

# vLLM concurrency per server — lower to create real queue pressure
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"

# Resume from the best checkpoint from previous training (val_solve_rate=0.7252)
# --checkpoint-path is the file to LOAD from (when --resume-checkpoint is set).
# After loading, new saves go to ${CHECKPOINT_DIR}/inframind_${DATASET}_{run_id}.pt
CHECKPOINT_PATH="${CHECKPOINT_DIR}/inframind_math_20260219_112548_job25279835_best.pt"

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

# Verify resume checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Resume checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi
echo "  Resume checkpoint: ${CHECKPOINT_PATH} ($(du -h "${CHECKPOINT_PATH}" | cut -f1))"
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
echo "InfraMind Training (Resume from best checkpoint)"
echo "Dataset: ${DATASET}"
echo "Limit: ${LIMIT} train, ${VAL_LIMIT} val"
echo "Epochs: ${EPOCHS}, Patience: ${PATIENCE}"
echo "Budget: LogUniform(${BUDGET_MIN}, ${BUDGET_MAX})"
echo "vLLM max_num_seqs: ${VLLM_MAX_NUM_SEQS}"
echo "Reward: quality-first + effort mandate + quality predictor shaping"
echo "Arrival rates: ${ARRIVAL_RATES}"
echo "Concurrency: ${CONCURRENCY}"
echo "Resume checkpoint: ${CHECKPOINT_PATH}"
echo "Predictors: latency, length, quality"
echo "========================================="

CMD="python Experiments/train_inframind_${DATASET}.py \
  --epochs ${EPOCHS} \
  --limit ${LIMIT} \
  --val-limit ${VAL_LIMIT} \
  --budget-min ${BUDGET_MIN} \
  --budget-max ${BUDGET_MAX} \
  --arrival-rates ${ARRIVAL_RATES} \
  --concurrency ${CONCURRENCY} \
  --training-batch-size ${TRAINING_BATCH_SIZE} \
  --patience ${PATIENCE} \
  --lr-scheduler --lr-patience 3 --lr-factor 0.5 \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --resume-checkpoint \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --quality-predictor ${QUALITY_PREDICTOR} \
  --checkpoint-path ${CHECKPOINT_PATH} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
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
