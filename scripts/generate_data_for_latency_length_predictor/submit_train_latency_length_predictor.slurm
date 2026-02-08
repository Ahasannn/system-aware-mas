#!/bin/bash
#SBATCH --job-name=train_predictors
#SBATCH --output=logs/generate_data_for_latency_length_predictor/slurm-train-%j.out
#SBATCH --error=logs/generate_data_for_latency_length_predictor/slurm-train-%j.err
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# Print job info
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Purpose: Train latency & length predictors"
echo "========================================="
echo ""

# Repository root
REPO_ROOT="/home/ah872032.ucf/system-aware-mas"
cd "$REPO_ROOT" || exit 1

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || exit 1
echo "Python: $(which python)"
python --version
echo ""

# Set environment variables
export TOKENIZERS_PARALLELISM="false"

# Ensure log directory exists
mkdir -p logs/generate_data_for_latency_length_predictor

# Run training
bash scripts/generate_data_for_latency_length_predictor/train_latency_length_predictor.sh
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "========================================="

exit $EXIT_CODE
