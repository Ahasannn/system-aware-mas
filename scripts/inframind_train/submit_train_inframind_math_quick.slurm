#!/bin/bash
#SBATCH --job-name=inframind_math_quick
#SBATCH --output=logs/inframind_training/math/slurm-quick-%j.out
#SBATCH --error=logs/inframind_training/math/slurm-quick-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Quick validation run: 519 items x 2 rates x 2 budgets x 1 epoch = ~2076 episodes
# Purpose: verify bug fixes — lambda stability, model distribution, quality retention
# Expected wall time: ~2.5 hours (rate=10 configs ~55 min each, rate=100 ~20 min each)

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

echo "Loading HPC environment configuration..."
source scripts/setup_hpc_env.sh

echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

echo "Blue Storage: ${BLUE_STORAGE}"
echo "HF Cache: ${HF_HOME}"
echo ""

mkdir -p logs/inframind_training/math

# Cleanup vLLM on exit
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

# Start vLLM
echo "Starting vLLM model pool..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready."
echo ""
bash scripts/check_vllm_status.sh
echo ""

# ===== Paths =====
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ah872032.ucf}"
MAS_CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth"
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"

CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/inframind"
mkdir -p "${CHECKPOINT_DIR}"

# To resume from a previous run, set this to the checkpoint path:
RESUME_CHECKPOINT=""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

echo "========================================="
echo "InfraMind QUICK VALIDATION — MATH"
echo "  Items:   519"
echo "  Epochs:  1"
echo "  Rates:   10 (idle), 100 (loaded)"
echo "  Budgets: 20s (tight), 150s (generous)"
echo "  Configs: 4 (shuffled)"
echo "  Total:   ~2076 episodes"
echo "========================================="
echo ""

CMD="python Experiments/train_inframind_math.py \
  --dataset-root ${MATH_DATASET_ROOT} \
  --limit 519 \
  --epochs 1 \
  --max-tokens 4096 \
  --arrival-rates 10,100 \
  --arrival-pattern poisson \
  --budget-sweep 20,150 \
  --concurrency 1000 \
  --training-batch-size 32 \
  --latency-predictor ${LATENCY_PREDICTOR} \
  --length-predictor ${LENGTH_PREDICTOR} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --checkpoint-every 100"

if [[ -f "${MAS_CHECKPOINT}" ]]; then
    echo "Loading MAS planner weights: ${MAS_CHECKPOINT}"
    CMD="${CMD} --mas-checkpoint ${MAS_CHECKPOINT}"
else
    echo "WARNING: MAS checkpoint not found: ${MAS_CHECKPOINT}"
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
TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Quick validation completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

exit $TRAIN_EXIT_CODE
