#!/bin/bash
#SBATCH --job-name=inframind_step2_judge
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step2_judge/slurm-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step2_judge/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# =========================================================================
# Step 2: LLM-as-a-Judge — score each agent response with 1-10 rubric
#         Uses Qwen2.5-72B-Instruct on 2x B200 with tensor-parallel
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

# Load HPC env (sets HF_HOME, TORCH_HOME, TMPDIR, etc.)
source scripts/setup_hpc_env.sh
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo "HF_HOME: ${HF_HOME}"
echo ""

# Load CUDA if needed
if ! command -v nvcc &> /dev/null; then
    module load cuda/12.8.1
fi

# --- Configuration ---
DATASET="${DATASET:-math}"
LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step2_judge"
mkdir -p "${STEP_DIR}"

INPUT_CSV="${INPUT_CSV:-${LOG_BASE}/step1_explore/explore.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${STEP_DIR}/scored.csv}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
JUDGE_PORT="${JUDGE_PORT:-8010}"
JUDGE_URL="http://localhost:${JUDGE_PORT}/v1"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# --- Deploy judge model via vLLM (following serve_full_pool.sh pattern) ---
JUDGE_LOG="${STEP_DIR}/vllm_judge_${SLURM_JOB_ID}.log"
JUDGE_PID_FILE="${STEP_DIR}/vllm_judge_${SLURM_JOB_ID}.pid"

cleanup_judge() {
    echo ""
    echo "Cleaning up judge vLLM server..."
    if [ -f "$JUDGE_PID_FILE" ]; then
        pid=$(cat "$JUDGE_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping judge (PID $pid)"
            kill -TERM "$pid" 2>/dev/null
            sleep 2
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
        fi
        rm -f "$JUDGE_PID_FILE"
    fi
    echo "Judge cleanup complete"
}
trap cleanup_judge EXIT INT TERM

echo "Starting judge model: ${JUDGE_MODEL} on port ${JUDGE_PORT}..."
echo "  Tensor Parallel Size: 2 (2x B200 — Qwen2.5-72B ~72GB/GPU)"
echo "  Max Model Len: 19000"
echo "  Log: ${JUDGE_LOG}"

# Use VLLM_USE_V1=0 for stability (matching serve_full_pool.sh)
VLLM_USE_V1=0 nohup python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${JUDGE_PORT}" \
    --model "${JUDGE_MODEL}" \
    --served-model-name "${JUDGE_MODEL}" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 19000 \
    --gpu-memory-utilization 0.70 \
    --no-enable-prefix-caching \
    --swap-space 16 \
    > "${JUDGE_LOG}" 2>&1 &
echo $! > "$JUDGE_PID_FILE"
echo "Judge PID: $(cat "$JUDGE_PID_FILE")"

# Wait for judge to be ready (up to 10 min for large model loading)
echo "Waiting for judge model to start..."
for i in $(seq 1 36000); do
    # Check if process died
    if ! kill -0 "$(cat "$JUDGE_PID_FILE")" 2>/dev/null; then
        echo "ERROR: Judge process died during startup. Check ${JUDGE_LOG}"
        tail -20 "${JUDGE_LOG}"
        exit 1
    fi
    if curl -s "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1; then
        echo "Judge model ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: Judge model failed to start within 600s. Check ${JUDGE_LOG}"
    tail -30 "${JUDGE_LOG}"
    exit 1
fi

# --- Run judge scoring ---
echo ""
echo "========================================="
echo "Running LLM-as-a-Judge scoring..."
echo "Input: ${INPUT_CSV}"
echo "Output: ${OUTPUT_CSV}"
echo "Judge: ${JUDGE_MODEL}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "========================================="

CMD="python Experiments/judge_responses.py \
  --input-csv ${INPUT_CSV} \
  --output-csv ${OUTPUT_CSV} \
  --judge-model ${JUDGE_MODEL} \
  --judge-url ${JUDGE_URL} \
  --max-concurrent ${MAX_CONCURRENT} \
  --resume"

echo "Command: $CMD"
set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Step 2 completed with exit code: $EXIT_CODE"
echo "Output CSV: ${OUTPUT_CSV}"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
