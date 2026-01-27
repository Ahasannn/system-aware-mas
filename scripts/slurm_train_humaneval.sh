#!/bin/bash
#SBATCH --job-name=mas_train_humaneval
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm_train_humaneval_%j.out
#SBATCH --error=logs/slurm_train_humaneval_%j.err

# Exit on error
set -e

echo "============================================"
echo "SLURM Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

# Change to project directory
cd /home/ji757406.ucf/ahasan/system-aware-mas

# Load CUDA module
echo "[$(date)] Loading CUDA module..."
module load cuda/12.8.1
echo "[$(date)] CUDA loaded: $(nvcc --version | head -1)"
echo ""

# Activate virtual environment
echo "[$(date)] Activating virtual environment..."
source .venv/bin/activate
echo "[$(date)] Python: $(which python)"
echo ""

# Create logs directory
mkdir -p logs

# Fix PyTorch cache directory permissions issue
# Use job-specific scratch directory or blue storage for caches
export TORCHINDUCTOR_CACHE_DIR="/blue/qi855292.ucf/ji757406.ucf/torch_cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/blue/qi855292.ucf/ji757406.ucf/triton_cache/${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"
echo "[$(date)] PyTorch cache: ${TORCHINDUCTOR_CACHE_DIR}"
echo "[$(date)] Triton cache: ${TRITON_CACHE_DIR}"
echo ""

# ==============================================================================
# STEP 1: Start vLLM servers
# ==============================================================================
echo "============================================"
echo "[$(date)] STEP 1: Starting vLLM servers..."
echo "============================================"
echo ""

bash scripts/vllm/serve_full_pool.sh > logs/vllm_startup_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "[$(date)] vLLM startup script launched (PID: $VLLM_PID)"
echo "[$(date)] Monitoring logs/vllm_startup_${SLURM_JOB_ID}.log"
echo ""

# ==============================================================================
# STEP 2: Wait for all models to be healthy
# ==============================================================================
echo "============================================"
echo "[$(date)] STEP 2: Waiting for models to be healthy..."
echo "============================================"
echo ""

PORTS=(8001 8002 8003 8004 8005)
MAX_WAIT=1200  # 20 minutes timeout
ELAPSED=0

all_healthy=false
while [ $ELAPSED -lt $MAX_WAIT ]; do
    all_healthy=true

    for port in "${PORTS[@]}"; do
        if ! curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            all_healthy=false
            echo "[$(date)] Port ${port} not ready yet..."
            break
        fi
    done

    if [ "$all_healthy" = true ]; then
        echo ""
        echo "[$(date)] ✓ All models are healthy!"
        break
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [ "$all_healthy" = false ]; then
    echo ""
    echo "[$(date)] ERROR: Models did not become healthy within ${MAX_WAIT} seconds"
    echo "[$(date)] Check logs/vllm_startup_${SLURM_JOB_ID}.log for details"
    exit 1
fi

echo ""
echo "Model endpoints ready:"
for port in "${PORTS[@]}"; do
    model=$(curl -s "http://127.0.0.1:${port}/v1/models" | head -1)
    echo "  Port ${port}: ${model}"
done
echo ""

# ==============================================================================
# STEP 3: Run training
# ==============================================================================
echo "============================================"
echo "[$(date)] STEP 3: Starting HumanEval training..."
echo "============================================"
echo ""

# Export environment variables needed by training script
export BLUE_STORAGE="/blue/qi855292.ucf/ji757406.ucf"
export HUMANEVAL_DATASET_PATH="${BLUE_STORAGE}/datasets/humaneval"
export KEY="EMPTY"
export PYTHONDONTWRITEBYTECODE=1

# Run training script directly (not in background, so we can monitor it)
bash scripts/mas_train_humaneval.sh 2>&1 | tee logs/train_humaneval_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================"
echo "[$(date)] Training completed with exit code: ${TRAIN_EXIT_CODE}"
echo "============================================"
echo ""

# ==============================================================================
# STEP 4: Cleanup - Stop vLLM servers
# ==============================================================================
echo "============================================"
echo "[$(date)] STEP 4: Stopping vLLM servers..."
echo "============================================"
echo ""

# Stop all vLLM servers
if [ -f scripts/vllm/stop_pool.sh ]; then
    bash scripts/vllm/stop_pool.sh
else
    # Fallback: kill by PID files
    for pidfile in logs/vllm/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            echo "[$(date)] Stopping process $pid (from $pidfile)"
            kill "$pid" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "============================================"
echo "SLURM Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "============================================"

exit $TRAIN_EXIT_CODE
