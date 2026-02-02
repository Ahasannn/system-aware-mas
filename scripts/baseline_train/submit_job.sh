#!/bin/bash
# Helper script to submit and monitor the SLURM job

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

SLURM_SCRIPT="scripts/baseline_train/submit_mas_train_math.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

echo "Submitting SLURM job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor job status with:"
echo "  squeue -j $JOB_ID"
echo ""
echo "View output logs:"
echo "  tail -f logs/baseline_mas_training/math/slurm-${JOB_ID}.out"
echo "  tail -f logs/baseline_mas_training/math/slurm-${JOB_ID}.err"
echo ""
echo "Cancel job:"
echo "  scancel $JOB_ID"
echo ""

# Save job ID for reference
echo "$JOB_ID" > .last_job_id
echo "Job ID saved to .last_job_id"
