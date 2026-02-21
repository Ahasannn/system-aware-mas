#!/bin/bash
#SBATCH --job-name=test_im_math
#SBATCH --output=logs/testing/math/slurm-%j-inframind.out
#SBATCH --error=logs/testing/math/slurm-%j-inframind.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# ===== InfraMind — Test Sweep for MATH (max-num-seqs=32) =====
# Inference-only with default vLLM concurrency.
# Produces CSV for comparison against baseline.
#
# Arrival rates: [10, 30, 100, 200] req/min
# Budgets:       [10, 50, 200] seconds
# Test items:    1000
#
# Total: 4 rates × 3 budgets × 500 items = 6,000 episodes
#
# Output CSV:
#   logs/testing/math/inframind_math_test_maxseq8.csv

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Source HPC env (sets BLUE_STORAGE, HF_HOME, cache dirs, etc.)
source scripts/setup_hpc_env.sh

# Activate virtual environment
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export CODE_EXECUTION_TIMEOUT="10"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# Default vLLM concurrency
export VLLM_MAX_NUM_SEQS=32

# Ensure log directory exists
mkdir -p logs/testing/math

# ===== Run settings =====
TEST_LIMIT=500
CONCURRENCY=1000
BUDGET_SWEEP="10,50,200"
ARRIVAL_RATES_CSV="10,30,100,200"
ARRIVAL_PATTERN="poisson"

# ===== Paths =====
CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind/inframind_math_phase2.pt"
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"
QUALITY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/quality_estimator.pth"

# ===== Output =====
OUTPUT_CSV="logs/testing/math/inframind_math_test_maxseq32.csv"

# ===== Validation =====
for f in "$CHECKPOINT" "$LATENCY_PREDICTOR" "$LENGTH_PREDICTOR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "${MATH_DATASET_ROOT}/test" ]]; then
    echo "ERROR: MATH test dataset not found at ${MATH_DATASET_ROOT}/test"
    exit 1
fi

# ===== Cleanup function =====
cleanup_vllm() {
    echo ""
    echo "Cleaning up vLLM servers..."

    # Use job-specific log directory to avoid killing other jobs' vLLM servers
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
                    if kill -0 "$pid" 2>/dev/null; then
                        kill -KILL "$pid" 2>/dev/null
                    fi
                fi
                rm -f "$pidfile"
            fi
        done
    fi
    echo "Cleanup complete"
}
trap cleanup_vllm EXIT INT TERM

# ===== Start vLLM (with max-num-seqs=32) =====
echo "Starting vLLM model pool (VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS})..."
bash scripts/vllm/serve_full_pool.sh || { echo "ERROR: Failed to start vLLM"; exit 1; }
echo "vLLM servers ready!"
bash scripts/check_vllm_status.sh
echo ""

echo "========================================="
echo "InfraMind — MATH Test Sweep (maxseq=32)"
echo "========================================="
echo "Checkpoint:      ${CHECKPOINT}"
echo "Latency pred:    ${LATENCY_PREDICTOR}"
echo "Length pred:     ${LENGTH_PREDICTOR}"
echo "Quality pred:    ${QUALITY_PREDICTOR}"
echo "Test limit:      ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Arrival rates:   ${ARRIVAL_RATES_CSV}"
echo "Arrival pattern: ${ARRIVAL_PATTERN}"
echo "Budget sweep:    ${BUDGET_SWEEP}"
echo "Max-num-seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e  # Don't exit on error; we want cleanup to run

python -m MAR.InfraMind.training \
    --dataset math \
    --dataset-root "${MATH_DATASET_ROOT}" \
    --split test \
    --limit "${TEST_LIMIT}" \
    --epochs 1 \
    --max-tokens 4096 \
    --arrival-rates "${ARRIVAL_RATES_CSV}" \
    --arrival-pattern "${ARRIVAL_PATTERN}" \
    --budget-sweep "${BUDGET_SWEEP}" \
    --concurrency "${CONCURRENCY}" \
    --latency-predictor "${LATENCY_PREDICTOR}" \
    --length-predictor "${LENGTH_PREDICTOR}" \
    --quality-predictor "${QUALITY_PREDICTOR}" \
    --checkpoint-path "${CHECKPOINT}" \
    --resume-checkpoint \
    --skip-training \
    --telemetry-csv "${OUTPUT_CSV}"

INFRAMIND_EXIT=$?

set -e

# ===== Summary =====
echo "========================================="
echo "InfraMind Test Sweep Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Exit status: ${INFRAMIND_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $INFRAMIND_EXIT
