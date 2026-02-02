# SLURM Training Guide for MAS Router on MATH Dataset

## Quick Start

1. **Submit the training job:**
   ```bash
   bash scripts/baseline_train/submit_job.sh
   ```

2. **Monitor the job:**
   ```bash
   # Check job status
   squeue -u $USER

   # View live output
   tail -f logs/baseline_mas_training/math/slurm-<JOB_ID>.out

   # View live errors
   tail -f logs/baseline_mas_training/math/slurm-<JOB_ID>.err
   ```

3. **Cancel the job if needed:**
   ```bash
   scancel <JOB_ID>
   ```

## What the SLURM Script Does

The script ([submit_mas_train_math.slurm](submit_mas_train_math.slurm)) automatically:

1. **Starts vLLM Model Pool** - Calls `scripts/vllm/serve_full_pool.sh` to launch all models
2. **Waits for Ready** - serve_full_pool.sh includes health checking built-in
3. **Verifies Status** - Runs `scripts/check_vllm_status.sh` to confirm all servers are healthy
4. **Runs Training** - Executes the MAS router training on MATH dataset
5. **Auto-Cleanup** - Shuts down vLLM servers when done (even if job fails or is cancelled)

## Resource Allocation

The script requests:
- **Account**: qi855292.ucf
- **Partition**: hpg-b200
- **Time**: 18 hours
- **Memory**: 180GB RAM
- **CPUs**: 24 cores
- **GPUs**: 2x B200 GPUs

**Adjust these if needed** in [submit_mas_train_math.slurm](submit_mas_train_math.slurm):
```bash
#SBATCH --account=qi855292.ucf
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:2    # GPU type and count
#SBATCH --time=18:00:00      # Walltime
#SBATCH --mem=180G           # RAM
#SBATCH --cpus-per-task=24   # CPU cores
```

## Key Fixes Applied

### 1. Memory Management
- **Garbage collection** after every 10 batches
- **GPU cache clearing** to prevent VRAM accumulation
- **Limited parallelism** (max 4 concurrent graph executions instead of batch_size)

### 2. vLLM Management
- **Uses existing scripts** - Calls `scripts/vllm/serve_full_pool.sh` instead of reimplementing
- Automatic startup and health checking built into serve_full_pool.sh
- **GPU allocation** - Configured in `MAR/LLM/llm_profile_full.json`
- Proper cleanup on exit (via trap) - kills all processes in logs/vllm/*.pid
- Individual log files per server in `logs/vllm/`

### 3. Checkpointing
- Auto-saves every 10 batches to `/blue/qi855292.ucf/ji757406.ucf/checkpoints/mas_router/mas_math_train_full.pth`
- Auto-resumes from checkpoint if exists
- Uses atomic writes (tmp file + rename) to prevent corruption

### 4. Environment Setup
- Uses `source .venv/bin/activate` (as requested)
- 18-hour time limit with hpg-b200 partition

## Troubleshooting

### Job Appears "Stuck" at Low Percentage (COMMON ISSUE)
**Symptoms**: Job shows 8% after 8+ hours, appears frozen
**Actual Cause**: NOT stuck - just extremely slow due to missing `--train_limit`

**What's Happening**:
- Training on ALL 7,500 examples (no limit set)
- 7,500 ÷ 16 batch_size = **468 batches per epoch**
- Each batch takes ~80 seconds (MATH problems are complex)
- Total time for 2 epochs = **~21 hours**
- But time limit is only **10 hours** → job will be killed!

**Fix Applied** (in submit_mas_train_math.slurm):
```bash
# NOW INCLUDES:
--train_limit 500 \    # Only use 500 examples
```

**New timeline**:
- 500 ÷ 16 = 31 batches per epoch
- 31 × 2 epochs = 62 total batches
- 62 × 80 sec ≈ **1.4 hours** (fits in 10-hour limit!)

**If you have a running "stuck" job**:
```bash
# Cancel it (will be killed anyway at 10h mark)
scancel <JOB_ID>

# Submit the fixed version
sbatch scripts/baseline_train/submit_mas_train_math.slurm
```

**For full dataset training** (remove from script):
```bash
# Remove or comment out:
# --train_limit 500 \

# AND increase time limit:
#SBATCH --time=24:00:00    # Changed from 10:00:00
```

### Job Dies After ~45 Batches
**Cause**: Memory accumulation or time limit
**Fix**: Already applied - memory cleanup every 10 batches

### vLLM Servers Won't Start
**Cause**: Not enough GPU memory or ports already in use
**Fix**: Check logs in `logs/vllm/vllm_*.log`
```bash
# Check if ports are in use
netstat -tuln | grep -E "8001|8002|8003|8004|8005"

# Kill stuck processes
pkill -f vllm.entrypoints
```

### "ModuleNotFoundError" or Import Errors
**Cause**: Wrong Python environment
**Fix**: The script uses `source .venv/bin/activate` - ensure the venv exists:
```bash
# Check if venv exists
ls -la .venv/bin/activate

# If needed, recreate it
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen
```

### Job Pending Forever
**Cause**: Insufficient resources available
**Fix**: Check queue and reduce resource requests
```bash
# See why job is pending
squeue -j <JOB_ID> --start

# Reduce requirements in SLURM script if needed
```

## Monitoring During Training

### Check Memory Usage
```bash
# SSH to compute node
ssh <node_name>

# Check memory
htop

# Check GPU memory
nvidia-smi -l 1
```

### Check vLLM Server Status
While job is running:
```bash
# SSH to the compute node first
ssh $(squeue -j <JOB_ID> -h -o "%N")

# Check vLLM health
curl http://localhost:8001/health
curl http://localhost:8002/health
# ... etc
```

## Training Progress Files

- **Checkpoint**: `/blue/qi855292.ucf/ji757406.ucf/checkpoints/mas_router/mas_math_train_full.pth`
- **Telemetry**: `logs/baseline_mas_training/math/mas_train_math_full_last.csv`
- **SLURM Output**: `logs/baseline_mas_training/math/slurm-<JOB_ID>.out`
- **SLURM Errors**: `logs/baseline_mas_training/math/slurm-<JOB_ID>.err`
- **vLLM Logs**: `logs/vllm/vllm_<PORT>.log`

## Customizing Training

Edit the training parameters in [submit_mas_train_math.slurm](submit_mas_train_math.slurm):

```bash
CMD="$PYTHON Experiments/run_math.py \
  --epochs 1 \              # Number of epochs
  --batch_size 8 \          # Batch size (reduce if OOM)
  --lr 0.01 \               # Learning rate
  --test_limit 16 \         # Test set size (for validation)
  --cost_rate 700.0 \       # Cost-quality tradeoff
  --train-limit 1000 \      # ADD THIS to test with fewer examples
  ...
```

## Advanced: Running Without SLURM

If you want to run interactively (not recommended for full training):

```bash
# Start vLLM pool manually
bash scripts/vllm/serve_full_pool.sh

# In another terminal, run training
bash scripts/baseline_train/mas_train_math.sh

# Stop vLLM when done
bash scripts/vllm/stop_pool.sh
```

## Code Changes Made

1. **Experiments/run_math.py**:
   - Added `import gc`
   - Added memory cleanup every 10 batches (GPU cache + garbage collection)

2. **MAR/MasRouter/mas_router.py**:
   - Limited ThreadPoolExecutor to max 4 workers (prevents overwhelming system)

3. **New files**:
   - `scripts/baseline_train/submit_mas_train_math.slurm` - Main SLURM job script
   - `scripts/baseline_train/submit_job.sh` - Helper to submit and track job
   - This README

## Next Steps

1. Try submitting the job:
   ```bash
   bash scripts/baseline_train/submit_job.sh
   ```

2. Monitor for first 100 batches to ensure stability

3. If it runs successfully past batch 100, it should complete the full training

4. Check the checkpoint is being saved regularly

Good luck! The memory issues should be resolved now.
