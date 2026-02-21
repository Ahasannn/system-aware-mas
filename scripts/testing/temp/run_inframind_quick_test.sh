#!/bin/bash
# ===== Quick InfraMind Test — For Demo Comparison vs Baseline =====
# Assumes vLLM models are ALREADY running (no model deployment).
# Run from repo root inside an srun session with vLLM active.
#
# Sweeps: 2 arrival_rates × 2 budgets × 500 items = 2,000 episodes
# Baseline comparison: logs/testing/math/baseline_math_test_maxseq32.csv

set -euo pipefail

REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT"

source .venv/bin/activate
source scripts/setup_hpc_env.sh

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"
export MATH_DATASET_ROOT="${BLUE_STORAGE}/datasets/MATH"

# ===== Paths =====
CHECKPOINT="${BLUE_STORAGE}/checkpoints/inframind/inframind_math_phase2.pt"
LATENCY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/latency_estimator.pth"
LENGTH_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/length_estimator.pth"
QUALITY_PREDICTOR="${BLUE_STORAGE}/checkpoints/predictors/quality_estimator.pth"

# ===== Test Settings =====
TEST_LIMIT=500
CONCURRENCY=1000
ARRIVAL_RATES="50,100"       # Low + moderate load
BUDGET_SWEEP="20,200"        # Tight + generous budget
ARRIVAL_PATTERN="poisson"

OUTPUT_CSV="logs/testing/math/inframind_math_quick_test.csv"
mkdir -p logs/testing/math

echo "========================================="
echo "InfraMind Quick Test — Demo Comparison"
echo "========================================="
echo "Arrival rates:   ${ARRIVAL_RATES}"
echo "Budget sweep:    ${BUDGET_SWEEP}"
echo "Items:           ${TEST_LIMIT}"
echo "Concurrency:     ${CONCURRENCY}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "========================================="
echo ""

# Verify checkpoint exists
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

python -m MAR.InfraMind.training \
    --dataset math \
    --dataset-root "${MATH_DATASET_ROOT}" \
    --split test \
    --limit "${TEST_LIMIT}" \
    --epochs 1 \
    --arrival-rates "${ARRIVAL_RATES}" \
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

EXIT_CODE=$?

echo ""
echo "========================================="
echo "InfraMind Quick Test Complete"
echo "========================================="
echo "Output CSV:  ${OUTPUT_CSV}"
echo "Exit status: ${EXIT_CODE}"
echo "========================================="

exit $EXIT_CODE
