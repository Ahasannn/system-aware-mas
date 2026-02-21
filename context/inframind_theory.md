# InfraMind: Infrastructure-Aware Multi-Agent Orchestration

## 1. Problem Formulation

Existing multi-agent LLM routing approaches select collaboration topologies, roles, and models based solely on task semantics, ignoring the infrastructure state of the serving layer. This infrastructure-oblivious routing causes: (1) **load imbalance** — static model preferences create deep queues on preferred models while others sit idle; (2) **avoidable queuing latency** — requests wait on congested models when idle alternatives exist; (3) **wasted compute** — workflows assigned infeasible topologies exceed their latency budget, discarding all generated tokens; and (4) **missed quality opportunities** — idle capacity at low load is never invested in richer reasoning.

InfraMind addresses all four problems by making infrastructure state observable at every decision level. Given a query $q$, a model pool $\mathcal{M}$, and a per-query latency budget $\beta$, InfraMind learns to:
1. **Select a collaboration topology** $\tau \in \mathcal{T}$ and assign roles $\mathcal{R}$ — quality-driven structural planning.
2. **Route each role to an LLM** $m \in \mathcal{M}$ with a prompting strategy $\sigma \in \Sigma$ — infrastructure-aware resource allocation.
3. **Schedule requests** via Earliest-Deadline-First (EDF) priority at the serving layer — deadline-aware scheduling.

We formulate this as a Constrained Markov Decision Process (CMDP):

$$\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right] \quad \text{s.t.} \quad \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] \leq \beta$$

where $R$ is the task quality reward and $C$ is the latency cost.

---

## 2. Architecture Overview

InfraMind uses a **two-level hierarchy** separating structural decisions from resource allocation:

| Level | Observes | Decides | Operates |
|-------|----------|---------|----------|
| **Planner** | Query embedding $\mathbf{e}_q$ | Topology $\tau$, agent count $K$, roles $\mathcal{R}$ | Once at $t=0$ |
| **Executor** | Query + role + remaining budget + system state | Model $m$, strategy $\sigma$ per role | Per agent node |
| **EDF Scheduler** | Deadline $D_i = t_i^{\text{arr}} + \beta_i$ | Queue ordering | Per request at serving layer |

### 2.1 Planner (Quality-Driven, No Budget Awareness)

The Planner uses the MAS routing pipeline (VAE + GFusion cross-attention) to select topology $\tau$, agent count $K$, and roles $\mathcal{R}$ from a query embedding. The planner operates on **task semantics only** — it is purely quality-driven with no budget conditioning.

**Design rationale:** The planner decides *what reasoning structure* to use (e.g., debate, chain-of-thought, reflection). This is a semantic decision that should depend on task difficulty and domain, not on infrastructure constraints. Budget awareness is delegated entirely to the executor, which decides *how* to execute each step cheaply or richly.

The planner is initialized from a pretrained MAS Router checkpoint (transferring task_classifier, collab_mode, num_agents, role_selector weights). The LLM router from MAS is discarded — model/strategy selection is handled by the executor.

**Available topologies:** IO (single agent), CoT (chain-of-thought), Chain, Debate, Reflection, FullConnected.

### 2.2 Executor (Infrastructure-Aware)

For each role $r_k$, the Executor selects a model and prompting strategy based on a state that fuses task semantics with real-time infrastructure conditions:

$$s_{\text{exec}}^{(k)} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_{r_k} \,\|\, b_{\text{rem}} \,\|\, \mathbf{z}_{\text{sys}} \right]$$

where:
- $\mathbf{e}_q \in \mathbb{R}^{384}$ = query embedding (SentenceTransformer)
- $\mathbf{e}_{r_k} \in \mathbb{R}^{384}$ = role embedding
- $b_{\text{rem}} \in \mathbb{R}^1$ = remaining budget (seconds)
- $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{|\mathcal{M}| \times |\Sigma|}$ = predicted latency map

**Action-Conditional Latency Map.** The system state enumerates the predicted latency for every (model, strategy) pair under current conditions:

$$\mathbf{z}_{\text{sys}} = \bigoplus_{m \in \mathcal{M}} \left[ \hat{L}_{m,\sigma_1} \,\|\, \cdots \,\|\, \hat{L}_{m,\sigma_{|\Sigma|}} \right]$$

Each predicted latency is computed as:

$$\hat{L}_{m,\sigma} = \hat{T}_{m,\sigma}^{\text{TTFT}} + \hat{T}_{m,\sigma}^{\text{TPOT}} \cdot \hat{N}_{m,\sigma}^{\text{out}}$$

This gives the executor a complete latency "price list" of all possible actions.

**Architecture:** A shared 2-layer MLP backbone (with LayerNorm and ReLU) with three factorized heads:

$$\pi_m, \; \pi_\sigma = \text{Softmax}(\text{Head}_m(\mathbf{h})), \; \text{Softmax}(\text{Head}_\sigma(\mathbf{h})), \quad V(s) = \text{Head}_V(\mathbf{h})$$

**Model pool** (5 models on vLLM):
- DeepSeek-R1-Distill-Qwen-32B (strongest, slowest)
- Mistral-Small-24B-Instruct-2501
- Qwen2.5-Coder-14B-Instruct
- Llama-3.1-8B-Instruct
- Llama-3.2-3B-Instruct (weakest, fastest)

**Prompt strategies** (3 strategies):
- Flash: "Go straight to the answer. No reasoning."
- Concise: "Brief 2-3 key points, then answer."
- DeepThink: "Reason thoroughly step-by-step, verify result."

### 2.3 EDF Scheduling at the Serving Layer

Each query's deadline $D_i = t_i^{\text{arr}} + \beta_i$ is propagated to vLLM as an EDF priority. Requests closer to their deadline are served first, reducing tail-latency violations.

---

## 3. Latency Estimation

Three neural predictors construct the latency map:

| Predictor | Input Features | Output |
|-----------|---------------|--------|
| **TTFT** (time-to-first-token) | Prompt token count, $n_m^{\text{run}}$, $n_m^{\text{wait}}$, KV cache %, avg TPOT/TTFT/queue/inference | Prefill + queuing delay |
| **TPOT** (time-per-output-token) | Same system metrics | Decode-phase speed |
| **Output Length** | Query embedding $\mathbf{e}_q$, model/role/strategy embeddings | Predicted output tokens |

All predictors use MLPs with Softplus output activations to ensure non-negative predictions. Strategy conditioning is critical because prompting strategies (Flash, Concise, DeepThink) produce dramatically different output lengths.

---

## 4. Training

### 4.1 Two-Level Training

Training is separated for the two levels:

**Planner training: Quality-first REINFORCE with effort mandate.**

$$\text{utility} = \begin{cases} 1.0 - \min\left(0.5,\; 0.3 \cdot \max\left(0, \frac{L}{\beta} - 1.0\right)\right) & \text{if solved} \\ -1.0 + 0.3 \cdot \min\left(1, \frac{L}{\beta}\right) & \text{if wrong} \end{cases}$$

- Correct answers: utility $\in [0.50, 1.0]$ — always positive, even if over budget
- Wrong answers: utility $\in [-1.0, -0.7]$ — effort mandate rewards trying harder
- Advantage: $A_i = \frac{\text{utility}_i - \mu}{\sigma}$ (normalized: mean-subtracted AND divided by std)
- Loss: $\mathcal{L} = -\log\pi \cdot A + \mathcal{L}_{\text{task}} + 0.001 \cdot \mathcal{L}_{\text{VAE}}$

**Executor training: Quality-first Actor-Critic with dense shaping.**

$$\text{reward} = \begin{cases} 1.0 - \min\left(0.5,\; 0.3 \cdot \max\left(0, \frac{L}{\beta} - 1.0\right)\right) & \text{if solved} \\ -1.0 + 0.3 \cdot w_q & \text{if wrong, quality predictor available} \\ -1.0 + 0.3 \cdot \min\left(1, \frac{L}{\beta}\right) & \text{if wrong, fallback to effort} \end{cases}$$

- Actor loss: $-\log\pi \cdot \hat{A}$ with normalized advantages (zero mean, unit variance)
- Critic loss: $\text{MSE}(V(s), \text{reward})$
- Entropy bonus: $\alpha_H \cdot \mathcal{H}[\pi]$ (coefficient 0.10)
- Total: $\mathcal{L} = \mathcal{L}_{\text{actor}} + 0.5 \cdot \mathcal{L}_{\text{critic}} - 0.10 \cdot \mathcal{H}$

**Design rationale.** Three key properties:
1. *Lexicographic dominance*: Correct answers ALWAYS outrank wrong answers (min gap = 1.20).
2. *Effort mandate*: Wrong answers that used more budget (tried harder) are penalized less than wrong answers that gave up quickly. This prevents collapse to cheap/fast configurations.
3. *Dense shaping*: The quality predictor provides per-step gradient within wrong answers — "almost correct" gets -0.7, "garbage" gets -1.0 — enabling the executor to learn which (model, strategy) pairs produce better outputs even when the episode fails.

### 4.2 Budget Randomization

Each training item samples budget $\beta \sim \text{LogUniform}(5, 300)$ seconds. Items within the same training batch share the same budget (sampled per batch block) so advantage normalization compares items at the same budget level.

The LogUniform range [5, 300] was derived from Phase 1 data: idle workflows take 5-30s, heavy load workflows take 60-290s.

### 4.3 Training Loop Structure

Each "epoch" consists of 7 arrival rate sweeps: [10, 30, 50, 100, 150, 200, 300] req/min (shuffled). For each sweep:
1. All training items are processed with Poisson arrivals at the given rate
2. Every 64 episodes, a training batch update is performed
3. Checkpoints saved every 50 episodes

After each full epoch (7 sweeps), validation is run deterministically on held-out data. LR scheduling (ReduceLROnPlateau) and early stopping (patience=5) are based on validation solve rate.

### 4.4 Quality Predictor (Step-Level Credit Assignment)

A trained quality predictor (MLP on query+response embeddings + model/role/strategy features) predicts a judge score $w_q \in [0, 10]$ for each executor step. When the episode outcome is wrong ($\text{is\_solved} = 0$), the quality predictor provides dense shaping: $\text{reward} = -1.0 + 0.3 \cdot (w_q / 10)$. This gives the executor a per-step gradient even within failed episodes — steps with higher predicted quality get less penalty, enabling the policy to learn that better (model, strategy) pairs produce more valuable outputs.

### 4.5 Workflow Latency

The workflow executes in topological waves. Nodes within a wave run in parallel; waves execute sequentially:

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} L_k$$

---

## 5. Notation Reference

| Symbol | Description |
|--------|-------------|
| $q$, $\mathbf{e}_q$ | Input query, query embedding |
| $\mathcal{M}$, $m$ | Model pool (5 models), selected model |
| $\mathcal{T}$, $\tau$ | Topology set (6 topologies), selected topology |
| $\mathcal{R}$, $r_k$ | Role set, role at position $k$ |
| $\Sigma$, $\sigma$ | Strategy set $\{\text{Flash, Concise, DeepThink}\}$, selected strategy |
| $\beta$, $b_{\text{rem}}$ | Total latency budget, remaining budget |
| $\hat{L}_{m,\sigma}$ | Predicted latency for model $m$ with strategy $\sigma$ |
| $n_m^{\text{run}}$, $n_m^{\text{wait}}$ | Running / waiting requests on model $m$ |
| $\pi_{\text{plan}}$, $\pi_{\text{exec}}$ | Planner policy, executor policy |
| $V(s)$ | Learned value function (executor critic) |
| $D_i$ | Deadline of query $i$: $t_i^{\text{arr}} + \beta_i$ |
| $w_q$ | Quality predictor weight for executor credit assignment |
