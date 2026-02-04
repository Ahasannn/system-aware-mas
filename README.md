# INFRAMIND: Infrastructure-Aware Multi-Agent Orchestration

**INFRAMIND** is a system-aware routing framework for multi-agent systems (MAS) that dynamically orchestrates LLM collaboration based on real-time infrastructure metrics and resource constraints.

## 🎯 Overview

Traditional LLM routing approaches focus solely on task characteristics, ignoring the underlying infrastructure state. **INFRAMIND** introduces infrastructure-awareness into multi-agent orchestration, using a hierarchical Constrained Markov Decision Process (CMDP) to balance accuracy, latency, and cost under dynamic system loads.

### Key Contributions

- **System-Aware Routing**: Real-time monitoring of vLLM metrics (queue depth, KV cache usage, latency) to inform routing decisions
- **Hierarchical CMDP Architecture**: Two-level decision making with Planner (topology + role selection) and Executor (LLM + strategy routing)
- **Load-Adaptive Orchestration**: Dynamic adjustment of agent collaboration patterns based on infrastructure state
- **Comprehensive Evaluation**: Tested across 5 datasets (MATH, MBPP, GSM8K, HumanEval, MMLU) under various load conditions

## 🏗️ Architecture

INFRAMIND consists of two main components:

1. **System-Aware Router** (`MAR/SystemRouter/`): Hierarchical CMDP-based routing with infrastructure monitoring
2. **Multi-Agent Graph Framework** (`MAR/Graph/`): Flexible execution engine for collaborative LLM reasoning

### System-Aware Router

- **Planner**: Selects collaboration topology and role set at query arrival (t=0)
- **Executor**: Dynamically routes (LLM, strategy) per role during execution based on:
  - Query embeddings
  - Role embeddings
  - Remaining budget
  - System metrics (queue depth, cache usage, latency predictions)

### Infrastructure Monitoring

Real-time metrics collection from vLLM endpoints:
- Queue depth (running + waiting requests)
- KV cache usage
- Time-to-first-token (TTFT)
- Inter-token latency (ITL)
- End-to-end latency

## 🚀 Quick Start

### Environment Setup

```bash
# Create virtual environment (Python 3.11 recommended)
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen

# Optional: For local vLLM serving
uv sync --frozen --extra serve
```

### Configuration

1. **API Keys**: Copy `template.env` to `.env` and add your API keys
   ```bash
   URL=""  # LLM backend URL
   KEY=""  # API key
   ```

2. **Blue Storage** (HPC): Configure paths in `scripts/setup_hpc_env.sh`
   ```bash
   export BLUE_STORAGE="/blue/qi855292.ucf/ah872032.ucf"
   export HF_HOME="${BLUE_STORAGE}/huggingface_cache"
   ```

### Datasets

Download and organize datasets in the `Datasets/` directory:

```
Datasets/
├── MATH/
│   ├── test/
│   └── train/
├── MBPP/
├── gsm8k/
├── humaneval/
└── MMLU/data/
```

Most datasets auto-download from HuggingFace on first use.

### Local vLLM Model Pool

Start a local pool of 5 vLLM servers on different ports:

```bash
# Start all models (ports 8001-8005)
bash scripts/vllm/serve_full_pool.sh

# Check health status
bash scripts/check_vllm_status.sh

# Stop all models
bash scripts/vllm/stop_pool.sh
```

Model pool includes:
- Qwen2.5-Coder-7B-Instruct (port 8001)
- Qwen2.5-32B-Instruct (port 8002)
- Qwen2.5-3B-Instruct (port 8003)
- Qwen2.5-0.5B-Instruct (port 8004)
- Qwen2.5-1.5B-Instruct (port 8005)

## 🧪 Running Experiments

### System-Aware Router Training

Train the infrastructure-aware router on each dataset:

```bash
# MBPP dataset
python Experiments/train_system_router_mbpp.py

# GSM8K dataset
python Experiments/train_system_router_gsm8k.py --dataset-path Datasets/gsm8k/gsm8k.jsonl

# MATH dataset
python Experiments/train_system_router_math.py --dataset-root Datasets/MATH

# MMLU dataset
python Experiments/train_system_router_mmlu.py --dataset-root Datasets/MMLU/data

# HumanEval dataset
python Experiments/train_system_router_humaneval.py --dataset-path Datasets/humaneval/humaneval-py.jsonl
```

### Baseline MAS Router Training (Comparison)

For comparison, train the baseline MAS Router without infrastructure awareness:

```bash
# Using SLURM (recommended for HPC)
sbatch scripts/baseline_train/submit_mas_train_mbpp.slurm
sbatch scripts/baseline_train/submit_mas_train_gsm8k.slurm
sbatch scripts/baseline_train/submit_mas_train_math.slurm
sbatch scripts/baseline_train/submit_mas_train_humaneval.slurm
sbatch scripts/baseline_train/submit_mas_train_mmlu.slurm

# Or run directly
python Experiments/run_mbpp.py --epochs 2 --batch_size 32 --lr 0.01
```

### Arrival Rate Sweeps (Load Testing)

Test routers under various arrival rates (requests per minute):

```bash
# System-Aware Router
sbatch scripts/motivation_plot_generator_data/submit_baseline_mas_test_arrival_sweep_mbpp.slurm

# Sweeps test at: 2, 5, 100, 200, 300 req/min with Poisson arrival pattern
```

Results are saved to `logs/motivation_plot_generator_data/`.

### Generating Plots

```bash
# Generate motivation plots from sweep results
python visualization/generate_motivation_plots.py

# Or use Jupyter notebook
jupyter notebook visualization/motivation_plots.ipynb
```

## 📊 Project Structure

```
.
├── MAR/                            # Core framework
│   ├── SystemRouter/              # Infrastructure-aware routing (INFRAMIND)
│   │   ├── system_router.py      # Hierarchical CMDP implementation
│   │   ├── metrics_watcher.py    # Real-time vLLM metrics collection
│   │   └── training.py           # CMDP training loop
│   ├── MasRouter/                # Baseline MAS Router
│   ├── Graph/                    # Multi-agent execution framework
│   ├── Agent/                    # Base agent implementation
│   ├── LLM/                      # LLM interface layer
│   ├── Roles/                    # Domain-specific agent roles
│   └── Prompts/                  # Prompt templates
│
├── Experiments/                   # Training and evaluation scripts
│   ├── train_system_router_*.py  # System-aware router training
│   └── run_*.py                  # Baseline MAS Router training
│
├── Datasets/                      # Dataset loaders
│   ├── math_dataset.py
│   ├── mbpp_dataset.py
│   ├── gsm8k_dataset.py
│   ├── humaneval_dataset.py
│   └── mmlu_dataset.py
│
├── scripts/                       # Utility scripts
│   ├── baseline_train/           # SLURM training jobs
│   ├── motivation_plot_generator_data/  # Load testing scripts
│   ├── vllm/                     # vLLM server management
│   └── setup_hpc_env.sh          # HPC environment setup
│
└── visualization/                 # Plotting and analysis
    ├── generate_motivation_plots.py
    └── motivation_plots.ipynb
```

## 📈 Key Features

### 1. Infrastructure Monitoring

Real-time collection of system metrics from vLLM endpoints:
- Automatically polls `/metrics` endpoint
- Tracks per-model queue depths and latencies
- Maintains sliding window of historical metrics

### 2. Constrained MDP Formulation

**State Space**:
- Query embedding (768-dim)
- Role embedding (768-dim)
- Remaining budget (scalar)
- System metrics vector (queue depth, cache usage, latency predictions)

**Action Space**:
- LLM selection (5 models)
- Prompting strategy (Flash, Concise, DeepThink)

**Reward**:
- Accuracy (task-specific correctness)
- Cost penalty (token usage × model pricing)
- Latency penalty (infrastructure-aware)

### 3. Hierarchical Decision Making

- **Planner** (t=0): Selects topology (single-agent CoT, debate, hierarchical) and roles
- **Executor** (runtime): Dynamically assigns LLM+strategy per role based on current system state

### 4. Load-Adaptive Behavior

Under high load:
- Prefers smaller, faster models
- Uses concise prompting strategies
- Reduces agent collaboration complexity

Under low load:
- Leverages larger models for accuracy
- Uses deeper reasoning strategies
- Enables richer multi-agent collaboration

## 🎓 Academic Use

This repository is designed for research and academic use. When using INFRAMIND in your research, please:

1. Cite the original MAS Router baseline (see Acknowledgments)
2. Reference our work (citation details will be added upon publication)
3. Follow the Apache 2.0 license terms

## 🙏 Acknowledgments

This work builds upon the **MAS Router** framework as a baseline:

```bibtex
@misc{yue2025masrouter,
  title={MasRouter: Learning to Route LLMs for Multi-Agent Systems},
  author={Yanwei Yue and Guibin Zhang and Boyang Liu and Guancheng Wan and Kun Wang and Dawei Cheng and Yiyan Qi},
  year={2025},
  eprint={2502.11133},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.11133}
}
```

We also thank the following projects for their contributions:
- [MapCoder](https://github.com/Md-Ashraful-Pramanik/MapCoder) - Multi-agent code generation
- [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) - Agent orchestration patterns
- [vLLM](https://github.com/vllm-project/vllm) - Efficient LLM serving

## 📝 Citation

Citation information will be added upon publication.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research prototype. For production use, additional hardening and optimization may be required.
