#!/bin/bash
#SBATCH --job-name=dryrun_step1_explore
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step1_explore/dryrun_slurm-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step1_explore/dryrun_slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# =========================================================================
# Dry-run Step 1: Small-scale real exploration (real models, real metrics)
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
DATASET="math"
DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"
MAS_CHECKPOINT="${BLUE_STORAGE}/checkpoints/mas_router/mas_math_train_519_cost100.pth"
LIMIT=5
ARRIVAL_RATES="0,100"
BUDGET_SWEEP="60"
CONCURRENCY=1000
EPOCHS=1

LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step1_explore"
mkdir -p "${STEP_DIR}"
TELEMETRY_CSV="${STEP_DIR}/dryrun_explore_${SLURM_JOB_ID}.csv"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

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

# --- Small-scale random exploration (real models, real metrics) ---
echo "========================================="
echo "Small-scale random exploration (real inference)"
echo "Dataset: ${DATASET}"
echo "MAS checkpoint: ${MAS_CHECKPOINT}"
echo "Limit: ${LIMIT}"
echo "Arrival rates: ${ARRIVAL_RATES}"
echo "Budget sweep: ${BUDGET_SWEEP}"
echo "Output: ${TELEMETRY_CSV}"
echo "========================================="

CMD="python Experiments/train_inframind_${DATASET}.py \
  --random-exploration \
  --mas-checkpoint ${MAS_CHECKPOINT} \
  --epochs ${EPOCHS} \
  --limit ${LIMIT} \
  --arrival-rates ${ARRIVAL_RATES} \
  --budget-sweep ${BUDGET_SWEEP} \
  --concurrency ${CONCURRENCY} \
  --telemetry-csv ${TELEMETRY_CSV} \
  --dataset-root ${DATASET_ROOT}"

echo "Command: $CMD"
set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Dry-run step 1 completed with exit code: $EXIT_CODE"
echo "Output CSV: ${TELEMETRY_CSV}"

if [ $EXIT_CODE -eq 0 ] && [ -f "${TELEMETRY_CSV}" ]; then
    echo ""
    echo "--- CSV Summary ---"
    echo "Total lines: $(wc -l < "${TELEMETRY_CSV}")"
    echo ""
    echo "--- Record types ---"
    awk -F',' 'NR>1 {print $4}' "${TELEMETRY_CSV}" | sort | uniq -c
    echo ""
    echo "--- Model distribution (role_step rows) ---"
    awk -F',' 'NR>1 && $4=="role_step" {print $24}' "${TELEMETRY_CSV}" | sort | uniq -c | sort -rn
    echo ""
    echo "--- Strategy distribution (role_step rows) ---"
    awk -F',' 'NR>1 && $4=="role_step" {print $25}' "${TELEMETRY_CSV}" | sort | uniq -c | sort -rn
    echo ""
    echo "--- Check prompt_base populated ---"
    awk -F',' 'NR>1 && $4=="role_step" {if ($23 != "") count++} END {print "prompt_base populated: " count " rows"}' "${TELEMETRY_CSV}"
    echo ""
    echo "--- Check response populated ---"
    awk -F',' 'NR>1 && $4=="role_step" {if ($22 != "") count++} END {print "response populated: " count " rows"}' "${TELEMETRY_CSV}"
    echo ""
    echo "--- Sample metrics (first 3 role_step rows: model, strategy, latency, ttft, tpot, llm_running, llm_kv_cache) ---"
    awk -F',' 'NR>1 && $4=="role_step" {print $24, $25, $26, $31, $32, $33, $34, $35; if (++n>=3) exit}' "${TELEMETRY_CSV}"
fi

echo ""
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
