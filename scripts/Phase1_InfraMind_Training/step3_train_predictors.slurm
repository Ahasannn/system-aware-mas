#!/bin/bash
#SBATCH --job-name=inframind_step3_predictors
#SBATCH --output=logs/InfraMind_Phase_1_Training/math/step3_predictors/slurm-%j.out
#SBATCH --error=logs/InfraMind_Phase_1_Training/math/step3_predictors/slurm-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# =========================================================================
# Step 3: Train all 3 predictors (latency, length, quality) from scored CSV
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
DATASET="${DATASET:-math}"
LOG_BASE="logs/InfraMind_Phase_1_Training/${DATASET}"
STEP_DIR="${LOG_BASE}/step3_predictors"
mkdir -p "${STEP_DIR}"

INPUT_CSV="${INPUT_CSV:-${LOG_BASE}/step2_judge/scored.csv}"
SAVE_DIR="${SAVE_DIR:-${BLUE_STORAGE}/checkpoints/predictors}"
RECORD_TYPE="${RECORD_TYPE:-role_step}"
EPOCHS="${EPOCHS:-100}"
LENGTH_EPOCHS="${LENGTH_EPOCHS:-200}"
LATENCY_EPOCHS="${LATENCY_EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"

mkdir -p "${SAVE_DIR}"

export TOKENIZERS_PARALLELISM="false"

# --- Train all 3 predictors ---
echo "========================================="
echo "Training latency + length + quality predictors..."
echo "Input CSV: ${INPUT_CSV}"
echo "Save dir: ${SAVE_DIR}"
echo "Record type: ${RECORD_TYPE}"
echo "Epochs: ${EPOCHS} (latency: ${LATENCY_EPOCHS}, length: ${LENGTH_EPOCHS})"
echo "========================================="

CMD="python Experiments/train_latency_length_predictor.py \
  --csv ${INPUT_CSV} \
  --save-dir ${SAVE_DIR} \
  --record-type ${RECORD_TYPE} \
  --epochs ${EPOCHS} \
  --length-epochs ${LENGTH_EPOCHS} \
  --latency-epochs ${LATENCY_EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR}"

echo "Command: $CMD"
set +e
$CMD
EXIT_CODE=$?
set -e

echo ""
echo "========================================="
echo "Step 3 completed with exit code: $EXIT_CODE"
echo "Checkpoints saved to: ${SAVE_DIR}"
ls -la "${SAVE_DIR}"/*.pth 2>/dev/null || echo "No checkpoints found"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
