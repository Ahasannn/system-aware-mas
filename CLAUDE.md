# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MasRouter is a research framework for learning to route LLMs in multi-agent systems (MAS). It uses a cascaded controller network to determine collaboration mode, role allocation, and LLM routing to balance effectiveness and efficiency.

**Paper**: [MasRouter: Learning to Route LLMs for Multi-Agent Systems](https://arxiv.org/abs/2502.11133) (ACL 2025)

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

## Running Experiments

### MasRouter Training/Testing (Original Router)
```bash
python Experiments/run_mbpp.py              # MBPP dataset
python Experiments/run_gsm8k.py             # GSM8K dataset
python Experiments/run_humaneval.py         # HumanEval dataset
python Experiments/run_math.py              # MATH dataset
python Experiments/run_mmlu.py              # MMLU dataset
```

Key flags for `run_*.py`:
- `--train_limit N`, `--test_limit N`: Limit examples
- `--epochs N`, `--batch_size N`, `--lr RATE`: Training hyperparameters
- `--checkpoint PATH`: Load a router checkpoint
- `--save-checkpoint PATH`: Save training checkpoint
- `--train-telemetry-csv`, `--test-telemetry-csv`: Output telemetry
- `--arrival-rate RATE`: Request arrival rate for load testing

### System-Aware Router Training (CMDP-based)
```bash
python Experiments/train_system_router_mbpp.py
python Experiments/train_system_router_gsm8k.py --dataset-path Datasets/gsm8k/gsm8k.jsonl
python Experiments/train_system_router_humaneval.py --dataset-path Datasets/humaneval/humaneval-py.jsonl
python Experiments/train_system_router_math.py --dataset-root Datasets/MATH
python Experiments/train_system_router_mmlu.py --dataset-root Datasets/MMLU/data
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

- **MasRouter/**: Original MasRouter with VAE-based task classification, collaboration determination, agent number selection, and LLM routing. Uses `GFusion` (Graph Fusion Module) for cross-attention between embeddings.

- **SystemRouter/**: System-Aware Router using hierarchical CMDP (Constrained Markov Decision Process):
  - Planner: Selects topology + role set at t=0
  - Executor: Selects (LLM, strategy) per role during runtime based on query embedding, role embedding, remaining budget, and system metrics

- **Graph/**: Multi-agent execution framework. `Graph` manages nodes and their spatial/temporal connections for collaborative reasoning workflows.

- **Agent/**: Base `Agent` class extending `Node`. Handles LLM calls with prompt limiting, token management, and role-based behavior.

- **LLM/**: LLM interface layer. `gpt_chat.py` handles OpenAI-compatible API calls. Model profiles in `llm_profile_full.json` define vLLM port mappings and model capabilities.

- **Roles/**: Domain-specific role definitions (Code, Math, Commonsense). Each role has prompts and behaviors for multi-agent collaboration.

- **Prompts/**: Prompt templates for tasks, reasoning strategies, and output formatting.

### Experiment Scripts (`Experiments/`)

- `run_*.py`: Full MasRouter training/testing for each dataset
- `train_system_router_*.py`: System-Aware Router training (wraps `MAR/SystemRouter/training.py`)

### Dataset Loaders (`Datasets/`)

Each dataset has a `*_dataset.py` loader. Datasets should be placed in subdirectories:
```
Datasets/
├── gsm8k/gsm8k.jsonl
├── humaneval/humaneval-py.jsonl
├── mbpp/mbpp.jsonl
├── MATH/{test,train}/
└── MMLU/data/
```

## Key Concepts

- **Reasoning Profiles**: Define collaboration topologies (single-agent CoT, multi-agent debate, etc.)
- **Prompt Strategies**: Flash (minimal), Concise (balanced), DeepThink (detailed reasoning)
- **System Metrics**: The System-Aware Router monitors vLLM metrics (queue depth, KV cache usage, latency) to make routing decisions under load
