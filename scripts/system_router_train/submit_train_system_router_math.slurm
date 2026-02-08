#!/bin/bash
#SBATCH --job-name=sysrouter_math
#SBATCH --output=logs/system_router_training/math/slurm-%j.out
#SBATCH --error=logs/system_router_training/math/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=12:00:00
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
echo ""

# Create log directories
mkdir -p logs/system_router_training/math

# Function to cleanup vLLM servers on exit
cleanup_vllm() {
    echo ""
    echo "========================================="
    echo "Cleaning up vLLM servers..."
    echo "========================================="

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

    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    echo "Cleanup complete"
    echo "========================================="
}

# Register cleanup on script exit
trap cleanup_vllm EXIT INT TERM

# Start vLLM model pool
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

# Run training
set +e
bash scripts/system_router_train/train_system_router_math.sh
TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

exit $TRAIN_EXIT_CODE
