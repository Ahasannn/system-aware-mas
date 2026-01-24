# [ACL 2025] MasRouter: Learning to Route LLMs for Multi-Agent Systems

## 📰 News

- 🎉 Updates (2025-5-15) MasRouter is accpected to ACL 2025 Main!
- 🚩 Updates (2025-2-16) Initial upload to arXiv [PDF](https://arxiv.org/abs/2502.11133).


## 🤔 Why MasRouter?

**MasRouter** expands LLM routing to the multi-agent systems (MAS) *for the first time*. It leverages the powerful reasoning capabilities of LLM MAS, while also making it relatively cost-effective.

![intro](assets/intro.png)

## 👋🏻 Method Overview

**MasRouter** integrates all components of MAS into a unified routing framework. It employs collaboration mode determination, role allocation, and LLM routing through a cascaded controller network, progressively constructing a MAS that balances effectiveness and efficiency.

![pipeline](assets/pipeline.png)

## 🏃‍♂️‍➡️ Quick Start

### 🧰 Environment (uv)

This repo includes `pyproject.toml` + `uv.lock` for a reproducible setup. Recommended Python is 3.11 (see `.python-version`).

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen
# Optional (for local vLLM serving):
# uv sync --frozen --extra serve
```

### 📊 Datasets

Please download the  `GSM8K`,  `HumanEval`, `MATH`, `MBPP`, `MMLU` datasets and place it in the `Datasets` folder. The file structure should be organized as follows:
```
Datasets
└── gsm8k
    └── gsm8k.jsonl
└── humaneval
    └── humaneval-py.jsonl
└── MATH
    └── test
    └── train
└── mbpp
    └── mbpp.jsonl
└── MMLU
    └── data
```

### 🔑 Add API keys

Add API keys in `template.env` and change its name to `.env`. We recommend that this API be able to access multiple LLMs.
```python
URL = "" # the URL of LLM backend
KEY = "" # the key for API
```

### 🖥️ Local vLLM model pool (optional)

If you want to run a small open-source model pool locally (e.g., multiple vLLM OpenAI servers on different GPUs/ports), you can use the helper scripts in `scripts/vllm/`.

For convenience, MasRouter will automatically look for a per-model base URL mapping at:
- `config_test.json` (recommended); or
- `MAR/LLM/llm_profile_full.json` (contains model profiles and URLs); or
- `logs/vllm/model_base_urls.json` (legacy, written by `scripts/vllm/serve_pool.sh`).

So you typically do **not** need to manually `export MODEL_BASE_URLS=...` for local testing.

### 🐹 Run the code

The code below verifies the experimental results of the `mbpp` dataset.

```bash
python experiments/run_mbpp.py
```

### 🧭 System-Aware Router training

The System-Aware Router training scripts are now available for multiple datasets:

```bash
python Experiments/train_system_router_mbpp.py
python Experiments/train_system_router_gsm8k.py --dataset-path Datasets/gsm8k/gsm8k.jsonl
python Experiments/train_system_router_math.py --dataset-root Datasets/MATH
python Experiments/train_system_router_mmlu.py --dataset-root Datasets/MMLU/data
python Experiments/train_system_router_humaneval.py --dataset-path Datasets/humaneval/humaneval-py.jsonl
```

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:
```bibtex
@misc{yue2025masrouter,
      title={MasRouter: Learning to Route LLMs for Multi-Agent Systems}, 
      author={Yanwei Yue and Guibin Zhang and Boyang Liu and Guancheng Wan and Kun Wang and Dawei Cheng and Yiyan Qi},
      year={2025},
      eprint={2502.11133},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.11133}, 
}
```

## 🙏 Acknowledgement

Special thanks to the following repositories for their invaluable code and datasets:

- [MapCoder](https://github.com/Md-Ashraful-Pramanik/MapCoder)
- [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).
