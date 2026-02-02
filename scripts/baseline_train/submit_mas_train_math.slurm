#!/bin/bash
#SBATCH --job-name=mas_train_math
#SBATCH --output=logs/baseline_mas_training/math/slurm-%j.out
#SBATCH --error=logs/baseline_mas_training/math/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Print job info
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Repository root
REPO_ROOT="/home/ji757406.ucf/ahasan/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"

# Checkpoint and telemetry paths
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_full.pth"
TELEMETRY_DIR="logs/baseline_mas_training/math"
TELEMETRY_CSV="${TELEMETRY_DIR}/mas_train_math_full_last.csv"
mkdir -p logs "${TELEMETRY_DIR}"

# Dataset configuration
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

# Set environment variables for training
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
COST_RATE="${COST_RATE:-700.0}"

# Function to cleanup vLLM servers on exit
cleanup_vllm() {
    echo ""
    echo "========================================="
    echo "Cleaning up vLLM servers..."
    echo "========================================="

    # Kill all vLLM processes from PID files
    if ls logs/vllm/*.pid >/dev/null 2>&1; then
        for pidfile in logs/vllm/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                name=$(basename "$pidfile" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Stopping $name (PID $pid)"
                    kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
                fi
                rm -f "$pidfile"
            fi
        done
    fi

    # Extra safety: kill any remaining vLLM processes
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    echo "Cleanup complete"
    echo "========================================="
}

# Register cleanup on script exit
trap cleanup_vllm EXIT INT TERM

# Start vLLM model pool using existing script
echo "========================================="
echo "Starting vLLM model pool..."
echo "========================================="
bash scripts/vllm/serve_full_pool.sh || {
    echo "ERROR: Failed to start vLLM model pool"
    echo "Check logs in logs/vllm/*.log"
    exit 1
}

echo ""
echo "========================================="
echo "vLLM servers are ready!"
echo "========================================="
echo ""

# Verify servers are healthy
bash scripts/check_vllm_status.sh
echo ""

# Build training command
# Full dataset training (7500 examples, ~21 hours for 2 epochs)
CMD="python Experiments/run_math.py \
  --epochs 1 \
  --batch_size 16 \
  --lr 0.01 \
  --test_limit 16 \
  --cost_rate ${COST_RATE} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --train-telemetry-csv ${TELEMETRY_CSV} \
  --start-batch 130 \
  --save-checkpoint ${CHECKPOINT_PATH}"

# Add checkpoint resumption if exists
if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "Found existing checkpoint: ${CHECKPOINT_PATH}"
    echo "Resuming training..."
    CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
else
    echo "No existing checkpoint found. Starting fresh training..."
fi

# Run training
echo "========================================="
echo "Starting training..."
echo "========================================="
echo "Command: $CMD"
echo ""

# Run with proper error handling
set +e  # Don't exit on error, we want cleanup to run
$CMD
TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

# Cleanup will happen automatically via trap
exit $TRAIN_EXIT_CODE
