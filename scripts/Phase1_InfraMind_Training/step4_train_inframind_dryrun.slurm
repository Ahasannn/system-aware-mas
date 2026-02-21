#!/bin/bash
#SBATCH --job-name=inframind_dryrun
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step4_train/dryrun-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step4_train/dryrun-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-default
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

# =========================================================================
# Step 4 DRY RUN: Validate the full InfraMind training pipeline
# No GPU, no vLLM — uses mock responses/latencies to test the full
# planner → executor → trainer loop with all new features:
#   - Fixed lambda (no dual ascent)
#   - LogUniform budget randomization
#   - Validation + early stopping + best-model checkpoint
#   - LR scheduling
#   - Baseline-subtracted planner REINFORCE
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

LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step4_train"
mkdir -p "${STEP_DIR}"
TELEMETRY_CSV="${STEP_DIR}/dryrun_${SLURM_JOB_ID}.csv"
CHECKPOINT_DIR="${STEP_DIR}/dryrun_checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

export KEY="EMPTY"
export TOKENIZERS_PARALLELISM="false"

# --- Dry run training ---
echo "========================================="
echo "DRY RUN — InfraMind Training Pipeline"
echo "  --dry-run: mock LLM responses, no vLLM needed"
echo "  --limit 10, --val-limit 5, --epochs 3"
echo "  --budget-min 5 --budget-max 300 (LogUniform)"
echo "  --fixed-lambda 1.0"
echo "  --arrival-rates 0,50 (2 rates)"
echo "  --patience 2 (early stopping)"
echo "  --lr-scheduler (ReduceLROnPlateau)"
echo "========================================="
echo ""

CMD="python Experiments/train_inframind_math.py \
  --dry-run \
  --epochs 3 \
  --limit 10 \
  --val-limit 5 \
  --budget-min 5 \
  --budget-max 300 \
  --fixed-lambda 1.0 \
  --arrival-rates 0,50 \
  --concurrency 1000 \
  --training-batch-size 8 \
  --patience 2 \
  --lr-scheduler \
  --lr-patience 1 \
  --lr-factor 0.5 \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --telemetry-csv ${TELEMETRY_CSV} \
  --dataset-root ${DATASET_ROOT}"

echo "Command: $CMD"
echo ""
set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Dry run completed with exit code: $EXIT_CODE"
echo "Telemetry: ${TELEMETRY_CSV}"
echo "Checkpoints: ${CHECKPOINT_DIR}/"
if [ $EXIT_CODE -eq 0 ]; then
    echo "STATUS: PASS — pipeline is working, safe to run full training"
    echo ""
    echo "Checkpoint files:"
    ls -lh "${CHECKPOINT_DIR}"/*.pt 2>/dev/null || echo "  (none)"
    echo ""
    echo "Telemetry rows:"
    wc -l "${TELEMETRY_CSV}" 2>/dev/null || echo "  (none)"
else
    echo "STATUS: FAIL — check errors above before running full training"
fi
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
