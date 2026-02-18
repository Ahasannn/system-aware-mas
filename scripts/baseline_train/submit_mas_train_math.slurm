#!/bin/bash
#SBATCH --job-name=mas_train_math
#SBATCH --output=logs/baseline_mas_training/math/slurm-%j.out
#SBATCH --error=logs/baseline_mas_training/math/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=06:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
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
REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Load centralized HPC environment configuration
echo "Loading HPC environment configuration..."
source scripts/setup_hpc_env.sh

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# Blue storage configuration (imported from setup_hpc_env.sh)
echo "Blue Storage: ${BLUE_STORAGE}"
echo "HF Cache: ${HF_HOME}"
echo "HF Token: $(if [[ -n "${HF_TOKEN}" ]]; then echo "✓ Configured"; else echo "✗ Missing"; fi)"
echo ""

# Read train_limit from dataset config
CONFIG_FILE="${REPO_ROOT}/Experiments/dataset_config.json"
TRAIN_LIMIT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['math']['train_limit'])")
echo "Train Limit (from config): ${TRAIN_LIMIT}"
echo ""

# Set environment variables for training
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
COST_RATE="${COST_RATE:-700.0}"

# Checkpoint and telemetry paths (include cost rate in name)
COST_TAG="cost$(printf '%.0f' ${COST_RATE})"
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mas_math_train_${TRAIN_LIMIT}_${COST_TAG}.pth"
TELEMETRY_DIR="logs/baseline_mas_training/math"
TELEMETRY_CSV="${TELEMETRY_DIR}/mas_train_math_${TRAIN_LIMIT}_${COST_TAG}.csv"
mkdir -p logs "${TELEMETRY_DIR}"

# Dataset configuration
MATH_DATASET_ROOT="${MATH_DATASET_ROOT:-${BLUE_STORAGE}/datasets/MATH}"
if [[ ! -d "${MATH_DATASET_ROOT}/train" || ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH dataset not found at ${MATH_DATASET_ROOT}"
    exit 1
fi

# Function to cleanup vLLM servers on exit
cleanup_vllm() {
    echo ""
    echo "========================================="
    echo "Cleaning up vLLM servers..."
    echo "========================================="

    # Use job-specific log directory
    local vllm_log_dir="logs/vllm/job_${SLURM_JOB_ID}"

    # Kill all vLLM processes from PID files in job-specific directory
    if ls "${vllm_log_dir}"/*.pid >/dev/null 2>&1; then
        for pidfile in "${vllm_log_dir}"/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                name=$(basename "$pidfile" .pid)
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Stopping $name (PID $pid)"
                    kill -TERM "$pid" 2>/dev/null
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null
                    fi
                fi
                rm -f "$pidfile"
            fi
        done
    else
        echo "No PID files found in ${vllm_log_dir}"
    fi

    # Extra safety: kill any vLLM processes spawned by this job
    # Only kill processes owned by current user to avoid affecting other jobs
    pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

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
# Training with config-specified limit (${TRAIN_LIMIT} examples)
CMD="python Experiments/run_math.py \
  --train_limit ${TRAIN_LIMIT} \
  --epochs 5 \
  --batch_size 32 \
  --lr 0.01 \
  --cost_rate ${COST_RATE} \
  --dataset-root ${MATH_DATASET_ROOT} \
  --train-telemetry-csv ${TELEMETRY_CSV} \
  --skip-test \
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
