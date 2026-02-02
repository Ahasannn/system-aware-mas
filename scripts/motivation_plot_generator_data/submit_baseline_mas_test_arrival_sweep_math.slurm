#!/bin/bash
#SBATCH --job-name=mas_sweep_math
#SBATCH --output=logs/motivation_plot_generator_data/slurm-%j.out
#SBATCH --error=logs/motivation_plot_generator_data/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=02:00:00
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

# Set environment variables
export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# Blue storage configuration
BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Ensure log directory exists
mkdir -p logs/motivation_plot_generator_data

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

# Run the arrival rate sweep
echo "========================================="
echo "Starting MATH arrival rate sweep..."
echo "========================================="
echo ""

set +e  # Don't exit on error, we want cleanup to run
bash scripts/motivation_plot_generator_data/baseline_mas_test_arrival_sweep_math.sh
SWEEP_EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Sweep completed with exit code: $SWEEP_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

# Cleanup will happen automatically via trap
exit $SWEEP_EXIT_CODE
