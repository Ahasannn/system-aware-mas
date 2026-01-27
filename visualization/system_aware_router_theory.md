# System-Aware Router: Theoretical Foundations

This document provides the formal mathematical foundations for the System-Aware Router, a hierarchical Constrained Markov Decision Process (CMDP) framework for intelligent LLM routing in Multi-Agent Systems.

---

## 1. Problem Formulation

### 1.1 Overview

Given a query $q$ and a pool of $M$ heterogeneous LLMs $\mathcal{M} = \{m_1, m_2, \ldots, m_M\}$ with varying capabilities and latencies, the System-Aware Router learns to:

1. **Select a collaboration topology** $\tau \in \mathcal{T}$ (e.g., single-agent, debate, ensemble)
2. **Assign roles** $\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$ based on the topology
3. **Route each role to an LLM** $m \in \mathcal{M}$ with a prompting strategy $\sigma \in \Sigma$

The objective is to **maximize task quality** while **satisfying latency constraints** under dynamic system conditions.

### 1.2 Constrained Markov Decision Process (CMDP)

We formulate the routing problem as a CMDP:

$$\mathcal{C} = \langle \mathcal{S}, \mathcal{A}, P, R, C, d_0, \gamma, \beta \rangle$$

where:
- $\mathcal{S}$: State space
- $\mathcal{A}$: Action space
- $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$: Transition probability
- $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$: Reward function
- $C: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}^+$: Cost function (latency)
- $d_0$: Initial state distribution
- $\gamma \in [0,1]$: Discount factor
- $\beta$: Latency budget constraint

### 1.3 Optimization Objective

The goal is to find a policy $\pi^*$ that maximizes expected cumulative reward subject to expected cumulative cost constraints:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

$$\text{subject to: } \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] \leq \beta$$

---

## 2. Hierarchical Architecture

The System-Aware Router employs a **two-level hierarchical structure**:

### 2.1 Planner (High-Level Policy)

At timestep $t=0$, the Planner selects the workflow configuration:

$$\pi_{\text{plan}}: \mathcal{S}_{\text{plan}} \rightarrow \Delta(\mathcal{T} \times \mathcal{R})$$

**Planner Action Space:**
$$a_{\text{plan}} = (\tau, \mathcal{R}) \in \mathcal{A}_{\text{plan}}$$

where:
- $\tau \in \mathcal{T} = \{\text{solo}, \text{chain}, \text{debate}, \text{ensemble}, \ldots\}$
- $\mathcal{R} \subseteq \mathcal{R}_{\text{all}}$ is the selected role set

### 2.2 Executor (Low-Level Policy)

For each role $r_k$ at step $t$, the Executor selects the LLM and strategy:

$$\pi_{\text{exec}}: \mathcal{S}_{\text{exec}} \rightarrow \Delta(\mathcal{M} \times \Sigma)$$

**Executor Action Space:**
$$a_{\text{exec}}^{(k)} = (m, \sigma) \in \mathcal{A}_{\text{exec}}$$

where:
- $m \in \mathcal{M}$: Selected LLM
- $\sigma \in \Sigma = \{\text{flash}, \text{concise}, \text{deep\_think}\}$: Prompting strategy

---

## 3. State Space Definitions

### 3.1 Planner State

The Planner observes the initial query state:

$$s_{\text{plan}} = \left[ \mathbf{e}_q \right] \in \mathbb{R}^{d_q}$$

where $\mathbf{e}_q = \text{Encoder}(q)$ is the query embedding from a pre-trained encoder.

### 3.2 Executor State

The Executor observes a rich state combining query, role, budget, and system information:

$$s_{\text{exec}}^{(k)} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_{r_k} \,\|\, b_{\text{rem}} \,\|\, \mathbf{z}_{\text{sys}} \right] \in \mathbb{R}^{d_{\text{exec}}}$$

where:
- $\mathbf{e}_q \in \mathbb{R}^{d_q}$: Query embedding
- $\mathbf{e}_{r_k} \in \mathbb{R}^{d_r}$: Role embedding (learnable)
- $b_{\text{rem}} \in \mathbb{R}$: Remaining latency budget
- $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{d_{\text{sys}}}$: System state vector

### 3.3 System State Vector

The system state captures the current load and estimated latency for each LLM:

$$\mathbf{z}_{\text{sys}} = \bigoplus_{m \in \mathcal{M}} \mathbf{z}_m$$

For each model $m$, the state vector includes:

$$\mathbf{z}_m = \left[ \hat{L}_m \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right]$$

where:
- $\hat{L}_m$: **Estimated latency** for model $m$ (see Section 4)
- $n_m^{\text{run}}$: Number of requests currently running
- $n_m^{\text{wait}}$: Number of requests waiting in queue

---

## 4. Latency Estimation

The latency estimation module predicts end-to-end request latency using neural network predictors for TTFT, TPOT, and output length.

### 4.1 End-to-End Latency Model

The estimated latency for a request to model $m$ is:

$$\hat{L}_m = \hat{T}_m^{\text{TTFT}} + \hat{T}_m^{\text{TPOT}} \cdot \hat{N}_m^{\text{out}}$$

where:
- $\hat{T}_m^{\text{TTFT}}$: Predicted Time-To-First-Token (prefill phase)
- $\hat{T}_m^{\text{TPOT}}$: Predicted Time-Per-Output-Token (decode phase)
- $\hat{N}_m^{\text{out}}$: Predicted output sequence length

### 4.2 TTFT Predictor

The Time-To-First-Token depends on input length and current system load. TTFT captures the time to process the input prompt (prefill) plus any queuing delay.

**Input Features:**
$$\mathbf{x}_{\text{TTFT}} = \left[ N^{\text{in}} \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right] \in \mathbb{R}^3$$

where:
- $N^{\text{in}}$: Input sequence length (number of tokens)
- $n_m^{\text{run}}$: Number of requests currently running on model $m$
- $n_m^{\text{wait}}$: Number of requests waiting in queue for model $m$

**MLP Architecture:**

$$\mathbf{h}_1^{\text{TTFT}} = \text{ReLU}\left( W_1^{\text{TTFT}} \mathbf{x}_{\text{TTFT}} + b_1^{\text{TTFT}} \right)$$

$$\mathbf{h}_2^{\text{TTFT}} = \text{ReLU}\left( W_2^{\text{TTFT}} \mathbf{h}_1^{\text{TTFT}} + b_2^{\text{TTFT}} \right)$$

$$\hat{T}_m^{\text{TTFT}} = \text{Softplus}\left( W_3^{\text{TTFT}} \mathbf{h}_2^{\text{TTFT}} + b_3^{\text{TTFT}} \right)$$

where:
- $W_1^{\text{TTFT}} \in \mathbb{R}^{d_h \times 3}$, $b_1^{\text{TTFT}} \in \mathbb{R}^{d_h}$
- $W_2^{\text{TTFT}} \in \mathbb{R}^{d_h \times d_h}$, $b_2^{\text{TTFT}} \in \mathbb{R}^{d_h}$
- $W_3^{\text{TTFT}} \in \mathbb{R}^{1 \times d_h}$, $b_3^{\text{TTFT}} \in \mathbb{R}$
- Softplus ensures non-negative output: $\text{Softplus}(x) = \log(1 + e^x)$

**Training Objective (MSE Loss):**

$$\mathcal{L}_{\text{TTFT}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{T}_m^{\text{TTFT},(i)} - T_m^{\text{TTFT},(i)} \right)^2$$

### 4.3 TPOT Predictor

The Time-Per-Output-Token captures decoding efficiency under concurrent load. TPOT increases with batch size due to memory bandwidth contention.

**Input Features:**
$$\mathbf{x}_{\text{TPOT}} = \left[ n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right] \in \mathbb{R}^2$$

**MLP Architecture:**

$$\mathbf{h}_1^{\text{TPOT}} = \text{ReLU}\left( W_1^{\text{TPOT}} \mathbf{x}_{\text{TPOT}} + b_1^{\text{TPOT}} \right)$$

$$\mathbf{h}_2^{\text{TPOT}} = \text{ReLU}\left( W_2^{\text{TPOT}} \mathbf{h}_1^{\text{TPOT}} + b_2^{\text{TPOT}} \right)$$

$$\hat{T}_m^{\text{TPOT}} = \text{Softplus}\left( W_3^{\text{TPOT}} \mathbf{h}_2^{\text{TPOT}} + b_3^{\text{TPOT}} \right)$$

where:
- $W_1^{\text{TPOT}} \in \mathbb{R}^{d_h \times 2}$, $b_1^{\text{TPOT}} \in \mathbb{R}^{d_h}$
- $W_2^{\text{TPOT}} \in \mathbb{R}^{d_h \times d_h}$, $b_2^{\text{TPOT}} \in \mathbb{R}^{d_h}$
- $W_3^{\text{TPOT}} \in \mathbb{R}^{1 \times d_h}$, $b_3^{\text{TPOT}} \in \mathbb{R}$

**Training Objective (MSE Loss):**

$$\mathcal{L}_{\text{TPOT}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{T}_m^{\text{TPOT},(i)} - T_m^{\text{TPOT},(i)} \right)^2$$

### 4.4 Output Length Predictor

The output length predictor estimates the number of tokens the model will generate, based on the query characteristics and prompting strategy.

**Input Features:**
$$\mathbf{x}_{\text{len}} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_\sigma \,\|\, \mathbf{e}_\tau \right] \in \mathbb{R}^{d_q + d_\sigma + d_\tau}$$

where:
- $\mathbf{e}_q \in \mathbb{R}^{d_q}$: Query embedding from pre-trained encoder
- $\mathbf{e}_\sigma \in \mathbb{R}^{d_\sigma}$: Learnable strategy embedding
- $\mathbf{e}_\tau \in \mathbb{R}^{d_\tau}$: Learnable topology embedding

**Strategy Embeddings:**
$$\mathbf{e}_\sigma = \text{Embedding}_{\text{strategy}}(\sigma) \quad \text{where } \sigma \in \{\text{flash}, \text{concise}, \text{deep\_think}\}$$

**MLP Architecture:**

$$\mathbf{h}_1^{\text{len}} = \text{ReLU}\left( W_1^{\text{len}} \mathbf{x}_{\text{len}} + b_1^{\text{len}} \right)$$

$$\mathbf{h}_2^{\text{len}} = \text{ReLU}\left( W_2^{\text{len}} \mathbf{h}_1^{\text{len}} + b_2^{\text{len}} \right)$$

$$\hat{N}_m^{\text{out}} = \text{Softplus}\left( W_3^{\text{len}} \mathbf{h}_2^{\text{len}} + b_3^{\text{len}} \right)$$

where:
- $W_1^{\text{len}} \in \mathbb{R}^{d_h \times (d_q + d_\sigma + d_\tau)}$
- Softplus ensures positive output length prediction

**Training Objective (MSE Loss):**

$$\mathcal{L}_{\text{len}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{N}_m^{\text{out},(i)} - N_m^{\text{out},(i)} \right)^2$$

### 4.5 Complete Latency Estimation Pipeline

**Forward Pass for Latency Estimation:**

```
Input: query q, strategy σ, topology τ, model m, system metrics (n_run, n_wait)

1. Encode query: e_q = Encoder(q)
2. Get input length: N_in = TokenCount(q)
3. Predict TTFT:
   x_TTFT = [N_in, n_m^run, n_m^wait]
   T̂_TTFT = MLP_TTFT(x_TTFT)
4. Predict TPOT:
   x_TPOT = [n_m^run, n_m^wait]
   T̂_TPOT = MLP_TPOT(x_TPOT)
5. Predict output length:
   x_len = [e_q || e_σ || e_τ]
   N̂_out = MLP_len(x_len)
6. Compute estimated latency:
   L̂_m = T̂_TTFT + T̂_TPOT × N̂_out

Output: L̂_m (estimated end-to-end latency)
```

### 4.6 System State Vector with Estimated Latency

The system state vector for the Executor incorporates estimated latencies:

$$\mathbf{z}_{\text{sys}} = \bigoplus_{m \in \mathcal{M}} \left[ \hat{L}_m \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right]$$

**Dimensionality:** For $M$ models, $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{3M}$

### 4.7 Joint Training of Latency Predictors

The latency predictors can be trained jointly with a combined loss:

$$\mathcal{L}_{\text{latency}} = \alpha_1 \mathcal{L}_{\text{TTFT}} + \alpha_2 \mathcal{L}_{\text{TPOT}} + \alpha_3 \mathcal{L}_{\text{len}}$$

where $\alpha_1, \alpha_2, \alpha_3$ are weighting coefficients (typically set to 1.0).

**End-to-End Latency Loss (Alternative):**

$$\mathcal{L}_{\text{E2E}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{L}_m^{(i)} - L_m^{(i)} \right)^2$$

where $L_m^{(i)}$ is the observed total latency for sample $i$.

---

## 5. Policy Networks

### 5.1 Planner Network Architecture

$$\pi_{\text{plan}}(a_{\text{plan}} | s_{\text{plan}}; \theta_{\text{plan}}) = \text{Softmax}\left( \text{MLP}_{\text{plan}}(\mathbf{e}_q) \right)$$

**Detailed Architecture:**

$$\mathbf{h}_{\text{plan}} = \text{ReLU}\left( W_1^{\text{plan}} \mathbf{e}_q + b_1^{\text{plan}} \right)$$
$$\mathbf{o}_{\text{plan}} = W_2^{\text{plan}} \mathbf{h}_{\text{plan}} + b_2^{\text{plan}}$$
$$\pi_{\text{plan}}(\cdot | s_{\text{plan}}) = \text{Softmax}(\mathbf{o}_{\text{plan}})$$

where $|\mathbf{o}_{\text{plan}}| = |\mathcal{T}| \times |\mathcal{P}(\mathcal{R}_{\text{all}})|$ covers all topology-roleset combinations.

### 5.2 Executor Network Architecture

$$\pi_{\text{exec}}(a_{\text{exec}} | s_{\text{exec}}; \theta_{\text{exec}}) = \pi_m(m | s_{\text{exec}}) \cdot \pi_\sigma(\sigma | s_{\text{exec}})$$

**Factorized Action Selection:**

**Model Head:**
$$\mathbf{h}_{\text{exec}} = \text{ReLU}\left( W_1^{\text{exec}} s_{\text{exec}} + b_1^{\text{exec}} \right)$$
$$\pi_m(\cdot | s_{\text{exec}}) = \text{Softmax}\left( W_m \mathbf{h}_{\text{exec}} + b_m \right)$$

**Strategy Head:**
$$\pi_\sigma(\cdot | s_{\text{exec}}) = \text{Softmax}\left( W_\sigma \mathbf{h}_{\text{exec}} + b_\sigma \right)$$

### 5.3 Role Embedding

Each role $r \in \mathcal{R}_{\text{all}}$ has a learnable embedding:

$$\mathbf{e}_r = \text{Embedding}_{\text{role}}(r) \in \mathbb{R}^{d_r}$$

---

## 6. Reward and Cost Functions

### 6.1 Quality Reward

The quality reward is computed based on task-specific evaluation:

$$R_{\text{quality}}(s, a) = Q(\text{response}, \text{ground\_truth})$$

For different tasks:
- **Code Generation:** $Q = \mathbb{1}[\text{pass\_all\_tests}]$
- **Math:** $Q = \mathbb{1}[\text{answer\_correct}]$
- **General QA:** $Q = \text{similarity}(\text{pred}, \text{gold})$

### 6.2 Latency Cost

The latency cost for the Executor at step $k$ is the observed execution time:

$$C_{\text{latency}}^{(k)} = L_k^{\text{observed}}$$

For the workflow, we use the **critical path latency** (accounting for parallelism):

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{k \in \text{wave}_w} L_k^{\text{observed}}$$

where $W$ is the number of sequential waves in the topology.

### 6.3 Budget Constraint

The remaining budget after step $k$ is:

$$b_{\text{rem}}^{(k+1)} = b_{\text{rem}}^{(k)} - L_k^{\text{observed}}$$

The constraint is satisfied when:

$$C_{\text{workflow}} \leq \beta$$

---

## 7. Lagrangian Relaxation

### 7.1 Lagrangian Formulation

We convert the constrained optimization to an unconstrained problem using the Lagrangian method:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right] - \lambda \left( \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] - \beta \right)$$

### 7.2 Dual Problem

The optimal policy solves the min-max problem:

$$\pi^* = \arg\max_{\pi} \min_{\lambda \geq 0} \mathcal{L}(\pi, \lambda)$$

### 7.3 Combined Reward

The effective reward used for policy gradient becomes:

$$\tilde{R}(s, a) = R_{\text{quality}}(s, a) - \lambda \cdot C_{\text{latency}}(s, a)$$

### 7.4 Lagrange Multiplier Update

The multiplier $\lambda$ is updated via gradient ascent on the dual:

$$\lambda \leftarrow \lambda + \eta_\lambda \left( \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} C(s_t, a_t) \right] - \beta \right)$$

**With Softplus Parameterization** (ensuring $\lambda \geq 0$):

$$\lambda = \text{softplus}(\tilde{\lambda}) = \log(1 + e^{\tilde{\lambda}})$$

$$\tilde{\lambda} \leftarrow \tilde{\lambda} + \eta_\lambda \cdot \text{constraint\_violation}$$

where:
$$\text{constraint\_violation} = C_{\text{workflow}} - \beta$$

---

## 8. Policy Gradient Training

### 8.1 REINFORCE with Baseline

For both Planner and Executor, we use REINFORCE with a baseline:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot (G - b(s)) \right]$$

where:
- $G$: Return (cumulative reward)
- $b(s)$: Baseline (running mean of returns)

### 8.2 Planner Policy Gradient

The Planner receives the episode-level return:

$$G_{\text{plan}} = R_{\text{quality}} - \lambda \cdot C_{\text{workflow}}$$

**Gradient:**
$$\nabla_{\theta_{\text{plan}}} J = \sum_{i=1}^{N} \nabla_{\theta_{\text{plan}}} \log \pi_{\text{plan}}(a_{\text{plan}}^{(i)} | s_{\text{plan}}^{(i)}) \cdot (G_{\text{plan}}^{(i)} - \bar{G}_{\text{plan}})$$

### 8.3 Executor Policy Gradient

Each Executor step receives a step-level return:

$$G_{\text{exec}}^{(k)} = R_{\text{quality}} - \lambda \cdot L_k^{\text{observed}}$$

**Gradient:**
$$\nabla_{\theta_{\text{exec}}} J = \sum_{i=1}^{N} \sum_{k=1}^{K_i} \nabla_{\theta_{\text{exec}}} \log \pi_{\text{exec}}(a_{\text{exec}}^{(i,k)} | s_{\text{exec}}^{(i,k)}) \cdot (G_{\text{exec}}^{(i,k)} - \bar{G}_{\text{exec}})$$

### 8.4 Complete Training Algorithm

**Algorithm: System-Aware Router Training**

```
Initialize: θ_plan, θ_exec, θ_TTFT, θ_TPOT, θ_len, λ̃ = 0, baseline_plan, baseline_exec
For each episode i = 1, ..., N:
    1. Encode query: e_q = Encoder(q_i)
       Compute input length: N_in = TokenCount(q_i)

    2. PLANNER STEP:
       s_plan = e_q
       Sample a_plan ~ π_plan(·|s_plan; θ_plan)
       (τ, R) = a_plan

    3. EXECUTOR STEPS (for each role r_k ∈ R):
       For k = 1, ..., K:
           Fetch system metrics: {n_m^run, n_m^wait}_{m∈M}
           For each model m ∈ M:
               T̂_m^TTFT = MLP_TTFT([N_in, n_m^run, n_m^wait]; θ_TTFT)
               T̂_m^TPOT = MLP_TPOT([n_m^run, n_m^wait]; θ_TPOT)
               N̂_m^out = MLP_len([e_q, e_σ, e_τ]; θ_len)
               L̂_m = T̂_m^TTFT + T̂_m^TPOT × N̂_m^out
           Compute system state: z_sys = [L̂_m, n_m^run, n_m^wait]_{m∈M}
           Compute remaining budget: b_rem = β - Σ_{j<k} L_j
           Assemble state: s_exec^(k) = [e_q || e_{r_k} || b_rem || z_sys]
           Sample a_exec^(k) ~ π_exec(·|s_exec^(k); θ_exec)
           Execute action, observe L_k, T_k^TTFT, T_k^TPOT, N_k^out

    4. EVALUATE:
       R_quality = Q(final_response, ground_truth)
       C_workflow = Σ_w max_{k∈wave_w} L_k
       λ = softplus(λ̃)

    5. COMPUTE RETURNS:
       G_plan = R_quality - λ · C_workflow
       G_exec^(k) = R_quality - λ · L_k

    6. UPDATE POLICIES (batch):
       θ_plan ← θ_plan + α · ∇_{θ_plan} Σ_i log π_plan · (G_plan - baseline_plan)
       θ_exec ← θ_exec + α · ∇_{θ_exec} Σ_{i,k} log π_exec · (G_exec - baseline_exec)

    7. UPDATE LATENCY PREDICTORS (optional, can be pre-trained):
       θ_TTFT ← θ_TTFT - α · ∇_{θ_TTFT} L_TTFT
       θ_TPOT ← θ_TPOT - α · ∇_{θ_TPOT} L_TPOT
       θ_len ← θ_len - α · ∇_{θ_len} L_len

    8. UPDATE LAGRANGE MULTIPLIER:
       λ̃ ← λ̃ + η_λ · (C_workflow - β)

    9. UPDATE BASELINES:
       baseline_plan ← (1-ρ) · baseline_plan + ρ · mean(G_plan)
       baseline_exec ← (1-ρ) · baseline_exec + ρ · mean(G_exec)
```

---

## 9. Workflow Latency Computation

### 9.1 Topological Execution

The workflow executes in waves based on the topology's DAG structure:

$$\text{wave}_1 = \{r : \text{in\_degree}(r) = 0\}$$
$$\text{wave}_{w+1} = \{r : \text{all predecessors in } \bigcup_{j \leq w} \text{wave}_j\}$$

### 9.2 Critical Path Latency

Since nodes within a wave execute in parallel, the workflow latency is:

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} L_k$$

### 9.3 Predicted Workflow Latency

Using the latency estimator, we can predict workflow latency before execution:

$$\hat{C}_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} \hat{L}_{m_k}$$

where $m_k$ is the LLM assigned to role $r_k$.

---

## 10. System Metrics Collection

### 10.1 Real-Time Metrics

The router continuously monitors each LLM server to capture system load:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Running Requests | $n_m^{\text{run}}$ | Active inference requests (prefill + decode) |
| Waiting Requests | $n_m^{\text{wait}}$ | Queued requests awaiting processing |

These metrics are collected via the vLLM `/metrics` endpoint at regular intervals.

### 10.2 Metrics for Latency Predictor Training

For training the latency predictors, we collect observed latency data:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Observed TTFT | $T_m^{\text{TTFT}}$ | Measured time-to-first-token |
| Observed TPOT | $T_m^{\text{TPOT}}$ | Measured time-per-output-token |
| Output Length | $N_m^{\text{out}}$ | Actual number of generated tokens |
| Input Length | $N^{\text{in}}$ | Number of input tokens |

### 10.3 Data Collection for Predictor Training

Training data is collected as tuples:

$$\mathcal{D}_{\text{TTFT}} = \left\{ \left( N^{\text{in},(i)}, n_m^{\text{run},(i)}, n_m^{\text{wait},(i)}, T_m^{\text{TTFT},(i)} \right) \right\}_{i=1}^{N}$$

$$\mathcal{D}_{\text{TPOT}} = \left\{ \left( n_m^{\text{run},(i)}, n_m^{\text{wait},(i)}, T_m^{\text{TPOT},(i)} \right) \right\}_{i=1}^{N}$$

$$\mathcal{D}_{\text{len}} = \left\{ \left( \mathbf{e}_q^{(i)}, \sigma^{(i)}, \tau^{(i)}, N_m^{\text{out},(i)} \right) \right\}_{i=1}^{N}$$

---

## 11. Theoretical Properties

### 11.1 Constraint Satisfaction

Under mild assumptions (bounded rewards/costs, ergodic MDP), the Lagrangian relaxation converges to a policy that satisfies:

$$\lim_{T \to \infty} \frac{1}{T} \mathbb{E}_{\pi^*} \left[ \sum_{t=0}^{T} C(s_t, a_t) \right] \leq \beta + \epsilon$$

for arbitrarily small $\epsilon > 0$.

### 11.2 Pareto Optimality

The learned policy achieves Pareto optimality on the quality-latency frontier:

$$\not\exists \pi' : Q(\pi') > Q(\pi^*) \text{ and } L(\pi') \leq L(\pi^*)$$

### 11.3 Adaptation to System Load

The system state $\mathbf{z}_{\text{sys}}$ enables dynamic adaptation:

$$\pi_{\text{exec}}(m | s_{\text{exec}}) \propto \exp\left( \text{value}(m) - \lambda \cdot \hat{L}_m \right)$$

Under high load ($n_m^{\text{run}} \uparrow$, $n_m^{\text{wait}} \uparrow$), the estimated latency $\hat{L}_m$ increases, causing the policy to shift probability mass to less loaded models with lower estimated latencies.

---

## 12. Summary of Key Equations

| Component | Equation |
|-----------|----------|
| **Latency Estimate** | $\hat{L}_m = \hat{T}_m^{\text{TTFT}} + \hat{T}_m^{\text{TPOT}} \cdot \hat{N}_m^{\text{out}}$ |
| **TTFT Prediction** | $\hat{T}_m^{\text{TTFT}} = \text{MLP}_{\text{TTFT}}([N^{\text{in}} \| n_m^{\text{run}} \| n_m^{\text{wait}}])$ |
| **TPOT Prediction** | $\hat{T}_m^{\text{TPOT}} = \text{MLP}_{\text{TPOT}}([n_m^{\text{run}} \| n_m^{\text{wait}}])$ |
| **Length Prediction** | $\hat{N}_m^{\text{out}} = \text{MLP}_{\text{len}}([\mathbf{e}_q \| \mathbf{e}_\sigma \| \mathbf{e}_\tau])$ |
| **Executor State** | $s_{\text{exec}} = [\mathbf{e}_q \| \mathbf{e}_r \| b_{\text{rem}} \| \mathbf{z}_{\text{sys}}]$ |
| **System State** | $\mathbf{z}_m = [\hat{L}_m \| n_m^{\text{run}} \| n_m^{\text{wait}}]$ |
| **Lagrangian Reward** | $\tilde{R} = R_{\text{quality}} - \lambda \cdot C_{\text{latency}}$ |
| **Constraint** | $C_{\text{workflow}} \leq \beta$ |
| **Multiplier Update** | $\lambda \leftarrow \lambda + \eta_\lambda (C_{\text{workflow}} - \beta)$ |
| **Policy Gradient** | $\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot (G - b)]$ |
| **Workflow Latency** | $C_{\text{workflow}} = \sum_w \max_{k \in \text{wave}_w} L_k$ |

---

## 13. Notation Reference

| Symbol | Description |
|--------|-------------|
| $q$ | Input query |
| $\mathbf{e}_q$ | Query embedding |
| $N^{\text{in}}$ | Input sequence length (tokens) |
| $\mathcal{M}$ | Set of available LLMs |
| $\mathcal{T}$ | Set of topologies |
| $\mathcal{R}$ | Set of roles |
| $\Sigma$ | Set of prompting strategies |
| $\tau$ | Selected topology |
| $m$ | Selected LLM |
| $\sigma$ | Selected strategy |
| $\beta$ | Latency budget |
| $b_{\text{rem}}$ | Remaining budget |
| $\lambda$ | Lagrange multiplier |
| $L$ | Observed latency |
| $\hat{L}$ | Predicted latency |
| $\hat{T}^{\text{TTFT}}$ | Predicted time-to-first-token |
| $\hat{T}^{\text{TPOT}}$ | Predicted time-per-output-token |
| $\hat{N}^{\text{out}}$ | Predicted output length |
| $n^{\text{run}}$ | Running requests |
| $n^{\text{wait}}$ | Waiting requests |
| $\pi_{\text{plan}}$ | Planner policy |
| $\pi_{\text{exec}}$ | Executor policy |
| $\theta$ | Policy parameters |
| $W$, $b$ | MLP weights and biases |
| $d_h$ | Hidden layer dimension |
| $G$ | Return |
| $b(s)$ | Baseline |

---

*This document serves as a theoretical reference for the System-Aware Router implementation in MasRouter.*
