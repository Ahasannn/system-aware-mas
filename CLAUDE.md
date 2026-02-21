# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**INFRAMIND** (Infrastructure-Aware Multi-Agent Orchestration) is a research framework for dynamically orchestrating LLM collaboration in multi-agent systems based on real-time infrastructure metrics and resource constraints.

Unlike traditional routing approaches that only consider task characteristics, INFRAMIND monitors vLLM infrastructure state (queue depth, KV cache usage, latency) and uses a hierarchical Constrained Markov Decision Process (CMDP) to adaptively balance accuracy, latency, and cost under dynamic system loads.

**Paper**: Citation details will be added upon publication

**Baseline**: This project uses [MasRouter](https://arxiv.org/abs/2502.11133) as a baseline comparison system.

## Build & Environment Setup

```bash
# Create and activate virtual environment (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen

# For local vLLM serving (optional)
uv sync --frozen --extra serve
```

**API Configuration**: Copy `template.env` to `.env` and add your API keys.

**HPC Configuration**: Edit `scripts/setup_hpc_env.sh` for blue storage paths and environment variables.

## Running Experiments

### System-Aware Router (INFRAMIND) - Main Contribution

Train the infrastructure-aware router with CMDP-based orchestration:

```bash
python Experiments/train_system_router_mbpp.py              # MBPP dataset
python Experiments/train_system_router_gsm8k.py --dataset-path Datasets/gsm8k/gsm8k.jsonl
python Experiments/train_system_router_humaneval.py --dataset-path Datasets/humaneval/humaneval-py.jsonl
python Experiments/train_system_router_math.py --dataset-root Datasets/MATH
python Experiments/train_system_router_mmlu.py --dataset-root Datasets/MMLU/data
```

Key flags for `train_system_router_*.py`:
- `--train_limit N`, `--test_limit N`: Limit examples
- `--epochs N`, `--batch_size N`, `--lr RATE`: Training hyperparameters
- `--checkpoint PATH`: Load a router checkpoint
- `--save-checkpoint PATH`: Save training checkpoint
- `--train-telemetry-csv`, `--test-telemetry-csv`: Output telemetry
- `--arrival-rate RATE`: Request arrival rate for load testing (requests per minute)

### Baseline MAS Router Training/Testing (For Comparison)

Train the original MAS Router without infrastructure awareness:

```bash
python Experiments/run_mbpp.py              # MBPP dataset
python Experiments/run_gsm8k.py             # GSM8K dataset
python Experiments/run_humaneval.py         # HumanEval dataset
python Experiments/run_math.py              # MATH dataset
python Experiments/run_mmlu.py              # MMLU dataset
```

**SLURM Training Jobs** (recommended for full training on HPC):
```bash
sbatch scripts/baseline_train/submit_mas_train_mbpp.slurm
sbatch scripts/baseline_train/submit_mas_train_gsm8k.slurm
sbatch scripts/baseline_train/submit_mas_train_humaneval.slurm
sbatch scripts/baseline_train/submit_mas_train_math.slurm
sbatch scripts/baseline_train/submit_mas_train_mmlu.slurm
```

**Arrival Rate Sweeps** (load testing):
```bash
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_mbpp.slurm
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_gsm8k.slurm
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_humaneval.slurm
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_math.slurm
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_mmlu.slurm
```

## Local vLLM Model Pool

Start the local model pool (5 models on ports 8001-8005):

```bash
bash scripts/vllm/serve_full_pool.sh    # Start all models
bash scripts/check_vllm_status.sh       # Check health status
bash scripts/vllm/stop_pool.sh          # Stop all models
```

Model URLs are auto-configured via `MAR/LLM/llm_profile_full.json`.

## Architecture

### Core Modules (`MAR/`)

#### **InfraMind/** - INFRAMIND Main Contribution

Infrastructure-aware routing using hierarchical CMDP:

- **inframind_router.py**: Hierarchical CMDP implementation
  - **Planner** (quality-driven): MAS-based VAE+GFusion pipeline. Selects topology + role set
    at t=0 based on query embedding only. No budget awareness — optimizes purely for quality.
  - **Executor** (infrastructure-aware): Selects (LLM, strategy) per role during runtime based on:
    - Query embedding + Role embedding
    - Remaining budget (adapts model/strategy choice to time pressure)
    - System metrics (queue depth, cache usage, predicted latencies per model×strategy)
  - Clean separation: planner decides WHAT reasoning structure to use, executor decides HOW to execute it cheaply.

- **metrics_watcher.py**: Real-time vLLM infrastructure monitoring
  - Polls `/metrics` endpoint on all model servers
  - Collects queue depth, KV cache usage, TTFT, ITL, E2E latency
  - Maintains sliding window of historical metrics
  - Provides state representation for executor

- **training.py**: Two-level training loop:
  - Planner: REINFORCE with normalized advantages, correct → [0.50, 1.0], wrong → [-1.0, -0.7] (effort mandate)
  - Executor: Actor-Critic with quality predictor shaping on wrong answers for dense credit assignment
  - Correct ALWAYS outranks wrong (min gap 1.20). Effort mandate: wrong+tried_hard > wrong+gave_up
  - LogUniform budget randomization per item for robust budget generalization
  - Validation, early stopping, best-model checkpointing, LR scheduling

#### **MasRouter/** - Baseline Comparison

Original MasRouter (VAE-based) without infrastructure awareness:
- Task classification via VAE
- Collaboration mode determination
- Agent number selection
- LLM routing based only on task features
- Uses `GFusion` (Graph Fusion Module) for cross-attention

#### **Graph/** - Multi-Agent Execution Framework

`Graph` manages nodes and their spatial/temporal connections for collaborative reasoning workflows. Supports multiple topologies: single-agent CoT, debate, hierarchical review, etc.

#### **Agent/** - Base Agent Implementation

`Agent` class extending `Node`. Handles:
- LLM calls with prompt limiting
- Token management
- Role-based behavior
- Response parsing

#### **LLM/** - LLM Interface Layer

- `gpt_chat.py`: OpenAI-compatible API calls
- Model profiles in `llm_profile_full.json` define vLLM port mappings and capabilities
- Automatic fallback to HuggingFace models

#### **Roles/** - Domain-Specific Role Definitions

Domain-specific role implementations:
- **Code/**: MBPP, HumanEval roles (coder, reviewer, debugger)
- **Math/**: MATH, GSM8K roles (problem solver, verifier)
- **Commonsense/**: MMLU roles (reasoner, validator)

Each role has prompts and behaviors for multi-agent collaboration.

#### **Prompts/** - Prompt Templates

Prompt templates for:
- Task-specific instructions
- Reasoning strategies (Flash, Concise, DeepThink)
- Output formatting
- Multi-agent communication

### Experiment Scripts (`Experiments/`)

- `train_system_router_*.py`: **INFRAMIND** training for each dataset
- `run_*.py`: Baseline MAS Router training/testing for comparison

### Dataset Loaders (`Datasets/`)

Each dataset has a `*_dataset.py` loader with stratified/deterministic sampling:

```
Datasets/
├── gsm8k_dataset.py       # GSM8K: math word problems
├── humaneval_dataset.py   # HumanEval: code generation
├── mbpp_dataset.py        # MBPP: code generation
├── MATH/                  # MATH: competition math problems
└── MMLU/                  # MMLU: multi-task language understanding
```

Datasets auto-download from HuggingFace or can be placed in subdirectories.

## Key Concepts

### Infrastructure-Aware Routing (INFRAMIND)

Unlike baseline MAS Router which only considers task features, INFRAMIND:

1. **Monitors System State**: Real-time vLLM metrics (queue depth, cache usage, latencies)
2. **Two-Level Decision Making**:
   - **Planner** (quality-driven): Selects topology + roles based on query semantics only
   - **Executor** (infrastructure-aware): Selects (model, strategy) per step based on remaining budget + system metrics
3. **Load-Adaptive Behavior** (executor-driven):
   - High load / tight budget → smaller models, Flash strategy
   - Low load / loose budget → larger models, DeepThink strategy
   - Topology/roles stay quality-optimal regardless of load
4. **Quality-First Reward with Effort Mandate**: Correctness is the primary objective. Correct → [+0.50, +1.0] (even if over budget). Wrong → [-1.0, -0.7] with effort mandate (tried harder = less penalty) and quality predictor dense shaping. Min gap 1.20 between worst correct and best wrong. Prevents collapse to cheap/fast configurations.
5. **LogUniform Budget Randomization**: Training samples budget ~ LogUniform(5, 300) per item for robust generalization across budget regimes.

### Reasoning Profiles

Define collaboration topologies:
- Single-agent Chain-of-Thought (CoT)
- Multi-agent Debate
- Hierarchical Review
- Parallel Verification

### Prompt Strategies

Granularity levels for reasoning:
- **Flash**: Minimal reasoning (fastest, cheapest)
- **Concise**: Balanced reasoning
- **DeepThink**: Detailed step-by-step reasoning (slowest, most accurate)

### System Metrics

INFRAMIND monitors:
- **Queue Depth**: Running + waiting requests per model
- **KV Cache Usage**: GPU memory utilization
- **TTFT**: Time to first token (queuing latency)
- **ITL**: Inter-token latency (generation speed)
- **E2E Latency**: End-to-end request time

## Development Guidelines

### Adding New Datasets

1. Create `Datasets/new_dataset.py` with loader class
2. Implement stratified/deterministic sampling with `limit` parameter
3. Add experiment script `Experiments/train_system_router_new_dataset.py`
4. Create SLURM scripts in `scripts/baseline_train/` and `scripts/motivation_plot_generator_data/`

### Adding New Roles

1. Create role class in `MAR/Roles/{Domain}/`
2. Define role prompts in `MAR/Prompts/`
3. Register role in experiment scripts
4. Update `MAR/Agent/reasoning_profile.py` if needed

### Modifying CMDP Architecture

Key files to modify:
- `MAR/InfraMind/inframind_router.py`: Planner (topology/role selection), executor (model/strategy MLP), reward computation
- `MAR/InfraMind/trainer.py`: Planner REINFORCE (quality-first), executor Actor-Critic, quality-first reward
- `MAR/InfraMind/training.py`: Training loop, budget randomization, validation, early stopping
- `MAR/InfraMind/metrics_watcher.py`: System state representation

## HPC Workflow

### Training Pipeline

1. **Setup**: `source scripts/setup_hpc_env.sh`
2. **Start vLLM**: `bash scripts/vllm/serve_full_pool.sh`
3. **Train System Router**: `python Experiments/train_system_router_mbpp.py`
4. **Train Baseline**: `sbatch scripts/baseline_train/submit_mas_train_mbpp.slurm`
5. **Load Testing**: `sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_mbpp.slurm`
6. **Generate Plots**: `python visualization/generate_motivation_plots.py`

### File Locations

**Checkpoints**: `/blue/qi855292.ucf/ah872032.ucf/checkpoints/`
- System-Aware Router: `system_router_{dataset}.pth`
- Baseline MAS Router: `mas_{dataset}_train_full.pth`

**Datasets**: `/blue/qi855292.ucf/ah872032.ucf/datasets/`

**Logs**: `logs/`
- Training telemetry: `logs/baseline_mas_training/{dataset}/`
- Sweep results: `logs/motivation_plot_generator_data/`
- vLLM logs: `logs/vllm/`

## Troubleshooting

### vLLM Issues
```bash
# Check server status
bash scripts/check_vllm_status.sh

# View logs
tail -f logs/vllm/*.log

# Restart servers
bash scripts/vllm/stop_pool.sh && bash scripts/vllm/serve_full_pool.sh
```

### SLURM Issues
```bash
# Check job queue
squeue -u $USER

# View job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>
```

### Memory Issues
- Reduce batch size in training scripts
- Adjust vLLM `--max-model-len` in `scripts/vllm/serve_full_pool.sh`
- Monitor GPU memory: `nvidia-smi -l 1`

## Important Notes

- **Research Prototype**: This is academic research code, not production-ready
- **Baseline Comparison**: MAS Router code is preserved in `MAR/MasRouter/` for comparison
- **Infrastructure Dependencies**: Requires vLLM servers with `/metrics` endpoint
- **Dataset Licensing**: Respect original dataset licenses when using this framework
