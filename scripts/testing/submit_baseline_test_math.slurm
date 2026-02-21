#!/bin/bash
#SBATCH --job-name=test_bl_math
#SBATCH --output=logs/testing/math/slurm-%j-baseline.out
#SBATCH --error=logs/testing/math/slurm-%j-baseline.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

# ===== Baseline MAS Router — Test Sweep for MATH (max-num-seqs=32) =====
# Sweeps cost_rate × arrival_rate with default vLLM concurrency.
#
# Cost rates:    [100]  (quick sanity test)
# Arrival rates: [10, 30, 100, 200] req/min
# Test items:    1000
#
# Total: 1 cost_rate × 4 arrival_rates × 500 items = 2,000 episodes
#
# Output CSV (single file, all cost_rates appended):
#   logs/testing/math/baseline_math_test_maxseq8.csv

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
ARRIVAL_RATES="10 30 100 200"
ARRIVAL_PATTERN="poisson"

# ===== Cost rates and corresponding checkpoints =====
COST_RATES=(100)
CHECKPOINT_DIR="${BLUE_STORAGE}/checkpoints/mas_router"

# ===== Output =====
OUTPUT_CSV="logs/testing/math/baseline_math_test_maxseq32.csv"

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

# ===== Validation =====
for cost in "${COST_RATES[@]}"; do
    CKPT="${CHECKPOINT_DIR}/mas_math_train_519_cost${cost}.pth"
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found: ${CKPT}"
        exit 1
    fi
done

echo "========================================="
echo "Baseline MAS Router — MATH Test Sweep (maxseq=32)"
echo "========================================="
echo "Cost rates:      ${COST_RATES[*]}"
echo "Arrival rates:   ${ARRIVAL_RATES}"
echo "Pattern:         ${ARRIVAL_PATTERN}"
echo "Test limit:      ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Max-num-seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

set +e  # Don't exit on error; we want cleanup to run

OVERALL_EXIT=0

for cost in "${COST_RATES[@]}"; do
    CHECKPOINT="${CHECKPOINT_DIR}/mas_math_train_519_cost${cost}.pth"

    for arrival_rate in $ARRIVAL_RATES; do
        echo "--- cost_rate=${cost}  arrival_rate=${arrival_rate}  pattern=${ARRIVAL_PATTERN} ---"

        python Experiments/run_math.py \
            --epochs 0 \
            --checkpoint "${CHECKPOINT}" \
            --cost_rate "${cost}" \
            --test_limit "${TEST_LIMIT}" \
            --concurrency "${CONCURRENCY}" \
            --arrival-rate "${arrival_rate}" \
            --arrival-pattern "${ARRIVAL_PATTERN}" \
            --test-telemetry-csv "${OUTPUT_CSV}"

        EXIT_CODE=$?
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "WARNING: cost_rate=${cost} arrival_rate=${arrival_rate} exited with code ${EXIT_CODE}"
            OVERALL_EXIT=1
        fi

        echo "Completed cost_rate=${cost} arrival_rate=${arrival_rate}"
        echo ""
    done
done

set -e

# ===== Summary =====
echo "========================================="
echo "Baseline Test Sweep Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Exit status: ${OVERALL_EXIT}"
echo "End Time:    $(date)"
echo "========================================="

exit $OVERALL_EXIT
