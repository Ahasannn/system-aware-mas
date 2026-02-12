# InfraMind: Theoretical Foundations

This document provides the formal mathematical foundations for InfraMind, a hierarchical Constrained Markov Decision Process (CMDP) framework for infrastructure-aware LLM routing in Multi-Agent Systems with Earliest-Deadline-First (EDF) serving-layer scheduling.

---

## 1. Problem Formulation

### 1.1 Motivation: Resource Underutilization in Infrastructure-Oblivious Routing

Existing multi-agent LLM routing approaches (e.g., MasRouter) select collaboration topologies, roles, and models based solely on **task characteristics** — the semantic content of the query. They are oblivious to the **infrastructure state**: queue depths, GPU memory pressure, and real-time latency dynamics of the serving layer. This obliviousness leads to three compounding forms of resource underutilization:

**Problem 1: Load Imbalance Across the Model Pool.**
Infrastructure-oblivious routers develop static preferences for certain models based on training-time quality correlations. Under concurrent load, these preferences create severe imbalance: preferred models accumulate deep queues while other models sit idle. Empirically, we observe queue depth disparities exceeding 10$\times$ between small and large model groups at high arrival rates (Figure 1a). The idle capacity of underutilized models represents wasted GPU resources that could serve requests faster.

**Problem 2: Avoidable Latency From Queue Congestion.**
As requests pile up on preferred models, queuing delay dominates end-to-end latency. Meanwhile, alternative models with empty queues could serve the same request with lower latency. The gap between the latency a request *experiences* on a congested model and the latency it *would have experienced* on an idle model is entirely avoidable (Figure 1b, shaded region). An infrastructure-aware router that senses queue depths can redirect requests to underutilized models, eliminating this queuing waste.

**Problem 3: Wasted Compute From Budget Violations.**
Every multi-agent workflow that exceeds its latency budget represents wasted compute: all tokens generated across all agents — prompt processing, generation, inter-agent communication — are discarded because the response arrived too late. An infrastructure-oblivious planner may assign a three-round debate topology to a query with a 10-second budget. Even if the executor picks the fastest models, the topology's inherent sequential structure (its **latency floor**) guarantees a budget violation. All compute spent on partial rounds is lost.

**Problem 4: Missed Quality Opportunities at Low Load.**
Conversely, when system load is low and budgets are generous, infrastructure-oblivious routers miss opportunities to *invest* idle capacity in richer reasoning. Large models that are idle could serve extended DeepThink prompts to boost quality, but a load-unaware system defaults to the same routing policy regardless of available headroom. This represents quality underutilization — the system could deliver better answers with the resources it already has.

**The Core Insight:** These four problems share a single root cause — the routing policy is **decoupled from infrastructure reality**. InfraMind addresses this by making infrastructure state observable at every decision level:

| Problem | Root Cause | InfraMind Solution |
|---------|------------|-------------------|
| Load imbalance | Static model preferences | Executor observes per-model queue depths, routes to underutilised models |
| Avoidable latency | Queue-blind routing | Action-conditional latency map predicts queuing delay for each (model, strategy) pair |
| Wasted compute | Budget-blind topology | Planner observes budget, selects structurally feasible topologies |
| Missed quality | Load-blind strategy | Executor shifts to DeepThink / larger models when system has headroom |
| FCFS queuing waste | Deadline-blind serving | EDF scheduling prioritises urgent requests, reducing budget violations at the serving layer |

### 1.2 Overview

Given a query $q$ and a pool of $M$ heterogeneous LLMs $\mathcal{M} = \{m_1, m_2, \ldots, m_M\}$ with varying capabilities and latencies, InfraMind learns to:

1. **Select a collaboration topology** $\tau \in \mathcal{T}$ (e.g., single-agent, debate, ensemble) — budget-aware structural adaptation
2. **Assign roles** $\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$ based on the topology
3. **Route each role to an LLM** $m \in \mathcal{M}$ with a prompting strategy $\sigma \in \Sigma$ — infrastructure-aware resource adaptation
4. **Schedule LLM requests** with Earliest-Deadline-First (EDF) priority at the serving layer — deadline-aware scheduling adaptation

The objective is to **maximize task quality** while **satisfying per-query latency deadlines** under dynamic system conditions, with the serving infrastructure co-operating via deadline-aware request scheduling. By coupling routing decisions to infrastructure state, InfraMind converts underutilised resources into quality improvements and eliminates wasted compute from budget violations.

### 1.3 Constrained Markov Decision Process (CMDP)

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

### 1.4 Optimization Objective

The goal is to find a policy $\pi^*$ that maximizes expected cumulative reward subject to expected cumulative cost constraints:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

$$\text{subject to: } \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] \leq \beta$$

---

## 2. Hierarchical Architecture

InfraMind employs a **three-level budget-aware hierarchy**, where each level directly addresses the resource underutilization problems identified in Section 1.1:

| Level | Observes | Decides | Adapts to | Solves |
|-------|----------|---------|-----------|--------|
| **Planner** | Query + total budget $\beta$ | Collaboration *structure* (topology, roles) | Task complexity under time constraint | Problem 3 (wasted compute from infeasible topologies) |
| **Executor** | Query + role + $b_{\text{rem}}$ + system state | *Resource* allocation per node (model, strategy) | Real-time infrastructure load | Problems 1, 2, 4 (load imbalance, queuing waste, missed quality) |
| **vLLM EDF** | Deadline $D_i = t_i^{\text{arr}} + \beta_i$ | Queue ordering | Request urgency | Problem 2 (avoidable latency from FCFS queuing) |

Budget awareness permeates every decision layer, from macro-structural choices down to queue scheduling. The hierarchy is designed so that each level handles the underutilisation problem at the appropriate granularity: the planner makes coarse structural decisions that determine the latency floor, the executor makes fine-grained per-node decisions that adapt to real-time system load, and the serving layer resolves contention among concurrent requests via deadline-aware scheduling.

### 2.1 Planner (High-Level Policy)

At timestep $t=0$, the Planner observes the query **and** the total latency budget to select the workflow configuration:

$$\pi_{\text{plan}}: \mathcal{S}_{\text{plan}} \rightarrow \Delta(\mathcal{T} \times \mathcal{R})$$

**Planner Action Space:**
$$a_{\text{plan}} = (\tau, K, \mathcal{R}) \in \mathcal{A}_{\text{plan}}$$

where:
- $\tau \in \mathcal{T} = \{\text{IO}, \text{CoT}, \text{Chain}, \text{FullConnected}, \text{Debate}, \text{Reflection}\}$
- $K \in \{1, \ldots, K_{\max}\}$ is the number of agents
- $\mathcal{R} = (r_1, \ldots, r_K)$ is the ordered role assignment

The planner makes these decisions through a **sequential pipeline** of four specialized modules — TaskClassifier, CollabDeterminer, NumDeterminer, and RoleAllocation — each backed by Variational Autoencoders (VAEs) and cross-attention (GFusion) over learned embeddings of topologies, roles, and task descriptions. This architecture, inherited from the MasRouter framework, learns rich latent representations of the combinatorial relationships between query difficulty, collaboration patterns, and role compositions.

**The gap in existing MAS planners** is that these modules operate on task semantics alone. They learn, for example, that competition-level algebra benefits from a three-agent debate, but they have no mechanism to consider whether a three-agent debate is *feasible* within the available time. A planner that assigns a Debate topology to a query with a 10-second budget has committed an irrecoverable error: the topology's sequential structure (its **latency floor**) guarantees a budget violation before the executor even begins.

**Budget-Conditioned Feature Modulation (BCFM).** InfraMind bridges this gap through a FiLM conditioning layer that modulates the query embedding *before* it enters the planner pipeline. Rather than redesigning the planner architecture (which would discard proven task-routing knowledge), BCFM transforms what the planner *perceives*:

$$\tilde{\mathbf{e}}_q = \gamma(\beta) \odot \mathbf{e}_q + \beta(\beta)$$

where $\gamma, \beta \in \mathbb{R}^{d_q}$ are scale and shift parameters produced by a budget-encoding MLP. This is Feature-wise Linear Modulation (FiLM), a well-established conditioning mechanism. The scale is initialized near identity ($\gamma \approx \mathbf{1}$) so that the modulation degrades gracefully to standard task-only behavior when budget information is uninformative.

**How BCFM steers structural decisions without modifying the planner.** Under tight budgets, BCFM learns to shift the query embedding toward regions of the latent space associated with simpler problems — causing the downstream CollabDeterminer to favor IO or CoT topologies, and the NumDeterminer to output fewer agents. Under generous budgets, the embedding is shifted toward the complex-problem region, enabling richer collaboration. The planner's internal modules remain architecturally identical to MasRouter; what changes is the input distribution they operate on.

**Transfer learning from pretrained MAS planners.** Because the planner modules share identical architectures with MasRouter, InfraMind initializes them from a pretrained MAS checkpoint. This transfers task-routing knowledge — which topologies and role combinations work for which problem types — without requiring InfraMind to re-learn it from scratch. The BCFM layer and executor are initialized randomly. End-to-end fine-tuning then adapts the planner from the MAS reward signal (token cost) to InfraMind's reward signal (latency/budget ratio), while preserving the task knowledge encoded in the VAE latent spaces.

### 2.2 Domain-Specific Role Sets

The planner selects from **domain-specific role sets** that define meaningful multi-agent collaborations. Each role set specifies the exact agents that will participate in the workflow. Role sets vary by domain to reflect domain expertise:

**Math Domain:**

| Role Set | Agents | Rationale |
|----------|--------|-----------|
| MathSolver | 1 agent | Fast single-expert for straightforward problems |
| MathSolver + Mathematician | 2 agents | Solver + verifier for moderate difficulty |
| MathSolver + Mathematician + MathTeacher | 3 agents | Multi-perspective for complex problems |
| MathSolver + Inspector | 2 agents | Solution + error checking |
| MathAnalyst + Mathematician + Inspector | 3 agents | Analysis + proof + verification |

**Code Domain:**

| Role Set | Agents | Rationale |
|----------|--------|-----------|
| ProgrammingExpert | 1 agent | Direct coding |
| ProgrammingExpert + TestAnalyst | 2 agents | Code + test review |
| ProjectManager + ProgrammingExpert + TestAnalyst | 3 agents | Plan + code + test |
| AlgorithmDesigner + ProgrammingExpert | 2 agents | Algorithm design + implementation |
| BugFixer + ProgrammingExpert | 2 agents | Debug + fix cycle |

The interplay between topology and role set determines the workflow structure: a Debate topology with a 3-agent role set produces 3 agents debating across 2 rounds, while Chain with 2 agents produces a sequential A→B pipeline. This combinatorial space is what the planner learns to navigate.

### 2.3 Executor (Low-Level Policy)

For each role $r_k$ at step $t$, the Executor selects the LLM and strategy:

$$\pi_{\text{exec}}: \mathcal{S}_{\text{exec}} \rightarrow \Delta(\mathcal{M} \times \Sigma)$$

**Executor Action Space:**
$$a_{\text{exec}}^{(k)} = (m, \sigma) \in \mathcal{A}_{\text{exec}}$$

where:
- $m \in \mathcal{M}$: Selected LLM
- $\sigma \in \Sigma = \{\text{Flash}, \text{Concise}, \text{DeepThink}\}$: Prompting strategy

---

## 3. State Space Definitions

### 3.1 Planner State

The Planner operates on a budget-conditioned query embedding produced by BCFM:

$$s_{\text{plan}} = \tilde{\mathbf{e}}_q = \text{BCFM}(\mathbf{e}_q, \beta_i) \in \mathbb{R}^{d_q}$$

where:
- $\mathbf{e}_q = \text{Encoder}(q) \in \mathbb{R}^{d_q}$ is the query embedding from a pre-trained sentence encoder
- $\beta_i$ is the per-query latency budget in seconds
- $\text{BCFM}(\mathbf{e}_q, \beta_i) = \gamma(\beta_i) \odot \mathbf{e}_q + \beta(\beta_i)$ applies FiLM conditioning

The budget-conditioned embedding $\tilde{\mathbf{e}}_q$ then flows through the sequential planner pipeline: TaskClassifier → CollabDeterminer → NumDeterminer → RoleAllocation. Each module operates on $\tilde{\mathbf{e}}_q$ (or context vectors derived from it) without any architectural modification — the budget signal enters entirely through BCFM's input modulation.

**Why budget modulation, not budget concatenation?** Concatenating the budget as an extra dimension ($[\mathbf{e}_q \| \beta_i] \in \mathbb{R}^{d_q+1}$) would require changing the input dimensions of all downstream modules, breaking compatibility with pretrained MAS weights. FiLM modulation preserves the embedding dimensionality ($\mathbb{R}^{d_q}$), enabling direct weight transfer from MAS while still providing the budget signal through a multiplicative pathway that the planner learns to interpret.

**Why budget, not system state?** The planner's job is *structural*: deciding how much collaboration a query needs. This depends on query difficulty and available time, not on per-model queue depths. System state changes between the planner's decision and agent execution; the executor, which runs per-node in real time, is the right level for infrastructure adaptation. Keeping the planner's budget interface minimal (a single scalar processed through BCFM) ensures the budget signal is not diluted by system noise.

### 3.2 Executor State

The Executor observes a rich state combining query, role, budget, and system information:

$$s_{\text{exec}}^{(k)} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_{r_k} \,\|\, b_{\text{rem}} \,\|\, \mathbf{z}_{\text{sys}} \right] \in \mathbb{R}^{d_{\text{exec}}}$$

where:
- $\mathbf{e}_q \in \mathbb{R}^{d_q}$: Query embedding
- $\mathbf{e}_{r_k} \in \mathbb{R}^{d_r}$: Role embedding (learnable)
- $b_{\text{rem}} \in \mathbb{R}$: Remaining latency budget
- $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{d_{\text{sys}}}$: System state vector

### 3.3 System State Vector (Action-Conditional Latency Map)

The system state captures the current load and **action-conditional estimated latencies** for each LLM. Rather than providing a single latency estimate per model (which would require knowing the strategy before it is selected), we enumerate the predicted latency for every candidate strategy, giving the executor a complete latency map of all possible actions under current infrastructure conditions.

$$\mathbf{z}_{\text{sys}} = \bigoplus_{m \in \mathcal{M}} \mathbf{z}_m$$

For each model $m$, the state vector includes the predicted latency for **each** strategy $\sigma \in \Sigma$, plus the live queue metrics:

$$\mathbf{z}_m = \left[ \hat{L}_{m,\sigma_1} \,\|\, \hat{L}_{m,\sigma_2} \,\|\, \cdots \,\|\, \hat{L}_{m,\sigma_{|\Sigma|}} \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right]$$

where:
- $\hat{L}_{m,\sigma}$: **Action-conditional estimated latency** for model $m$ with strategy $\sigma$ (see Section 4)
- $n_m^{\text{run}}$: Number of requests currently running
- $n_m^{\text{wait}}$: Number of requests waiting in queue

**Dimensionality:** For $M$ models and $|\Sigma|$ strategies, $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{M(|\Sigma|+2)}$

---

## 4. Latency Estimation

The latency estimation module predicts end-to-end request latency using neural network predictors for TTFT, TPOT, and output length.

### 4.1 End-to-End Latency Model

The estimated latency for a request to model $m$ under strategy $\sigma$ is:

$$\hat{L}_{m,\sigma} = \hat{T}_{m,\sigma}^{\text{TTFT}} + \hat{T}_{m,\sigma}^{\text{TPOT}} \cdot \hat{N}_{m,\sigma}^{\text{out}}$$

where:
- $\hat{T}_{m,\sigma}^{\text{TTFT}}$: Predicted Time-To-First-Token (prefill phase), conditioned on strategy $\sigma$
- $\hat{T}_{m,\sigma}^{\text{TPOT}}$: Predicted Time-Per-Output-Token (decode phase), conditioned on strategy $\sigma$
- $\hat{N}_{m,\sigma}^{\text{out}}$: Predicted output sequence length, conditioned on strategy $\sigma$

Strategy conditioning is critical because different prompting strategies (Flash, Concise, DeepThink) produce dramatically different output lengths (often 5-10x variation), which directly determines the generation-phase latency via the $\hat{T}^{\text{TPOT}} \cdot \hat{N}^{\text{out}}$ term.

### 4.2 TTFT Predictor

The Time-To-First-Token depends on input length and current system load. TTFT captures the time to process the input prompt (prefill) plus any queuing delay.

**Input Features:**
$$\mathbf{x}_{\text{TTFT}} = \left[ N^{\text{in}} \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right] \in \mathbb{R}^3$$

**MLP Architecture:**

$$\hat{T}_m^{\text{TTFT}} = \text{Softplus}\left( \text{MLP}_{\text{TTFT}}(\mathbf{x}_{\text{TTFT}}) \right)$$

where Softplus ensures non-negative output: $\text{Softplus}(x) = \log(1 + e^x)$.

### 4.3 TPOT Predictor

$$\hat{T}_m^{\text{TPOT}} = \text{Softplus}\left( \text{MLP}_{\text{TPOT}}([n_m^{\text{run}} \,\|\, n_m^{\text{wait}}]) \right)$$

### 4.4 Output Length Predictor

$$\hat{N}_m^{\text{out}} = \text{Softplus}\left( \text{MLP}_{\text{len}}([\mathbf{e}_q \,\|\, \mathbf{e}_\sigma \,\|\, \mathbf{e}_m]) \right)$$

### 4.5 Complete Latency Estimation Pipeline (Action-Conditional)

```
Input: query q, model pool M, strategy set Sigma, system metrics {n_m^run, n_m^wait}_{m in M}

1. Encode query: e_q = Encoder(q)
2. Get input length: N_in = TokenCount(q)
3. For each model m in M:
     For each strategy sigma in Sigma:
       a. L_hat(m,sigma) = TTFT_hat(m,sigma) + TPOT_hat(m,sigma) * N_hat_out(m,sigma)

Output: {L_hat(m,sigma)}_{m in M, sigma in Sigma} (complete latency map)
```

---

## 5. Policy Networks

### 5.1 Planner Network Architecture

The planner is a **sequential decision pipeline** of four VAE-based modules, preceded by Budget-Conditioned Feature Modulation. Each module makes one decision, and its output conditions the next module's input. All modules use learned latent spaces with temperature-scaled stochastic sampling for exploration.

**Step 0 — Budget Conditioning (BCFM):**

$$\mathbf{h}_\beta = \text{MLP}_\beta(\beta_i) \in \mathbb{R}^{d_h}$$
$$\gamma = 1 + W_\gamma \mathbf{h}_\beta, \quad \beta = W_\beta \mathbf{h}_\beta$$
$$\tilde{\mathbf{e}}_q = \gamma \odot \mathbf{e}_q + \beta$$

The scale is centered at identity ($\gamma \approx 1$) to preserve pretrained representations at initialization.

**Step 1 — Task Classification:**

$$\mathbf{z}_q = \text{VAE}_q(\tilde{\mathbf{e}}_q), \quad \mathbf{z}_t = \text{VAE}_t(\mathbf{e}_{\text{tasks}})$$
$$\pi_{\text{task}} = \text{Softmax}\left(\frac{\mathbf{z}_q \cdot \mathbf{z}_t^\top}{\tau_{\text{task}}}\right)$$

Produces the query context vector $\mathbf{c}_q = \text{Proj}(\tilde{\mathbf{e}}_q) \in \mathbb{R}^{d_h}$.

**Step 2 — Collaboration / Topology Selection (CollabDeterminer):**

$$\mathbf{z}_c = \text{VAE}_c(\mathbf{e}_{\text{collabs}}), \quad \mathbf{z}_{cq} = \text{VAE}_{cq}(\tilde{\mathbf{e}}_q)$$
$$\pi_\tau = \text{Softmax}\left(\frac{\mathbf{z}_{cq} \cdot \mathbf{z}_c^\top}{\tau_{\text{collab}}}\right)$$
$$\tau \sim \text{CDF-Sample}(\pi_\tau), \quad \log p_\tau = \log \pi_\tau[\tau]$$

The selected topology's latent vector $\mathbf{z}_c[\tau]$ becomes the collaboration context for role allocation.

**Step 3 — Agent Count (NumDeterminer):**

$$\hat{\mathbf{e}}_q, \mathbf{z}_n, \mu_n, \sigma_n = \text{VAE}_n(\tilde{\mathbf{e}}_q)$$
$$K = \text{clamp}\left(\text{round}\left(\text{sigmoid}(W_n \mathbf{z}_n) \cdot K_{\max}\right), 1, K_{\max}\right)$$

**Step 4 — Sequential Role Allocation (RoleAllocation):**

For each agent slot $j = 1, \ldots, K$, the module selects a role conditioned on the planner context and previously selected roles:

$$\mathbf{c}_{\text{plan}} = [\mathbf{c}_q \,\|\, \mathbf{z}_c[\tau]] \in \mathbb{R}^{2d_h}$$
$$\mathbf{z}_r = \text{VAE}_r(\mathbf{e}_{\text{roles}}), \quad \mathbf{h}_j = \text{GFusion}(\mathbf{c}_{\text{plan}}, \mathbf{h}_{j-1})$$
$$\pi_{r_j} = \text{Softmax}\left(\frac{\mathbf{h}_j \cdot \mathbf{z}_r^\top}{\tau_{\text{role}}}\right)$$
$$r_j \sim \text{CDF-Sample}(\pi_{r_j}), \quad \log p_{r_j} = \log \pi_{r_j}[r_j]$$

where $\mathbf{h}_0$ is a learned initial embedding and $\mathbf{h}_j$ accumulates the history of selected roles via layer-normalized addition.

**Combined planner log-probability and VAE loss:**

$$\log \pi_{\text{plan}} = \log p_\tau + \sum_{j=1}^{K} \log p_{r_j}$$
$$\mathcal{L}_{\text{VAE}} = \sum_{\text{module}} \left(\text{MSE}(\hat{x}, x) - \frac{1}{2}\text{KL}(q(z|x) \| p(z))\right)$$

The VAE reconstruction losses regularize the latent spaces of all modules, preventing mode collapse in the learned embeddings.

### 5.2 Executor Network Architecture

The executor uses factorized action selection with a shared backbone and separate heads for model and strategy:

$$\mathbf{h}_{\text{exec}} = \text{Backbone}_{\text{exec}}(s_{\text{exec}})$$
$$\pi_m(\cdot | s_{\text{exec}}) = \text{Softmax}(\text{Head}_m(\mathbf{h}_{\text{exec}}))$$
$$\pi_\sigma(\cdot | s_{\text{exec}}) = \text{Softmax}(\text{Head}_\sigma(\mathbf{h}_{\text{exec}}))$$
$$V(s_{\text{exec}}) = \text{Head}_V(\mathbf{h}_{\text{exec}})$$

The value head $V(s)$ provides the advantage baseline for the executor policy gradient.

---

## 6. Reward and Cost Functions

### 6.1 Quality Reward

The quality reward is computed based on task-specific evaluation:

$$R_{\text{quality}}(s, a) = Q(\text{response}, \text{ground\_truth})$$

For different tasks:
- **Code Generation:** $Q = \mathbb{1}[\text{pass\_all\_tests}]$
- **Math:** $Q = \mathbb{1}[\text{answer\_correct}]$
- **General QA:** $Q = \text{similarity}(\text{pred}, \text{gold})$

### 6.2 Latency Cost (Proportional)

The latency cost is the **normalized** observed latency, always providing a gradient signal regardless of whether the budget is met:

**Planner cost (episode-level):**
$$C_{\text{plan}} = \frac{C_{\text{workflow}}}{\beta}$$

**Executor cost (step-level):**
$$C_{\text{exec}}^{(k)} = \frac{L_k^{\text{observed}}}{b_{\text{rem}}^{(k)}}$$

where $b_{\text{rem}}^{(k)}$ is the remaining budget at step $k$.

**Why proportional and not one-sided?** A one-sided penalty $\max(0, C_{\text{workflow}} - \beta) / \beta$ produces zero gradient when under budget, removing all incentive to differentiate between actions of different latencies. The proportional cost ensures:

- Under budget ($C < \beta$): cost is in $[0, 1)$, providing gradient to prefer faster options
- At budget ($C = \beta$): cost is exactly $1.0$
- Over budget ($C > \beta$): cost grows linearly, penalizing violations

This continuous gradient is essential for learning meaningful topology and model preferences. Without it, the policy collapses to whichever action first achieves zero penalty (typically the cheapest single-agent option) regardless of quality.

### 6.3 Budget Constraint

The remaining budget after step $k$ is:

$$b_{\text{rem}}^{(k+1)} = b_{\text{rem}}^{(k)} - L_k^{\text{observed}}$$

The constraint is satisfied when:

$$C_{\text{workflow}} \leq \beta$$

---

## 7. Lagrangian Relaxation

### 7.1 Lagrangian Formulation

We convert the constrained optimization to an unconstrained problem using the Lagrangian method:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_{\pi} \left[ R_{\text{quality}} \right] - \lambda \left( \mathbb{E}_{\pi} \left[ \frac{C_{\text{workflow}}}{\beta} \right] - 1 \right)$$

The budget-normalized constraint $\mathbb{E}[C/\beta] \leq 1$ ensures the Lagrange multiplier operates on a consistent scale regardless of the absolute budget value.

### 7.2 Dual Problem

The optimal policy solves the min-max problem:

$$\pi^* = \arg\max_{\pi} \min_{\lambda \geq 0} \mathcal{L}(\pi, \lambda)$$

### 7.3 Combined Reward

The effective reward used for policy gradient becomes:

$$\tilde{R}_{\text{plan}} = R_{\text{quality}} - \lambda \cdot \frac{C_{\text{workflow}}}{\beta}$$

$$\tilde{R}_{\text{exec}}^{(k)} = R_{\text{quality}} - \lambda \cdot \frac{L_k}{b_{\text{rem}}^{(k)}}$$

### 7.4 Lagrange Multiplier Update (Signed Dual Ascent)

The multiplier $\lambda$ is updated via **signed** gradient ascent on the dual. This is critical for correct CMDP behavior:

$$\lambda \leftarrow \max\left(0, \; \lambda + \eta_\lambda \left( \frac{\bar{C}_{\text{workflow}}}{\bar{\beta}} - 1 \right) \right)$$

where $\bar{C}_{\text{workflow}}$ and $\bar{\beta}$ are the batch means of workflow latency and budget respectively.

**The signed update is essential.** When the average latency is below budget ($\bar{C}/\bar{\beta} < 1$), the constraint gap is negative, and $\lambda$ **decreases**. This relaxes the latency pressure, allowing the policy to explore more expensive but potentially higher-quality topologies (multi-agent debate, larger models, DeepThink strategies). When the average latency exceeds budget ($\bar{C}/\bar{\beta} > 1$), $\lambda$ increases, tightening the constraint.

**Failure mode of unsigned update:** If the update uses $\text{relu}(C - \beta)$ instead of the signed gap, $\lambda$ can only increase (monotonically). The policy becomes increasingly latency-averse over training, eventually collapsing to the cheapest possible action (single-agent, smallest model, Flash strategy) regardless of quality. This is a degenerate equilibrium that satisfies the constraint but maximizes neither quality nor the Lagrangian objective.

With Softplus parameterization (ensuring $\lambda \geq 0$):

$$\tilde{\lambda} \leftarrow \tilde{\lambda} + \eta_\lambda \cdot \left( \frac{\bar{C}_{\text{workflow}}}{\bar{\beta}} - 1 \right)$$

$$\lambda = \text{softplus}(\tilde{\lambda})$$

---

## 8. Policy Gradient Training

### 8.1 Entropy Regularization

Both the planner and executor policies include entropy regularization to prevent premature convergence and maintain exploration:

$$\mathcal{H}[\pi] = -\sum_a \pi(a|s) \log \pi(a|s)$$

The entropy bonus is added to the policy gradient objective:

$$J(\theta) = \mathbb{E}_\pi[\tilde{R} \cdot \log \pi(a|s)] + \alpha_H \cdot \mathbb{E}[\mathcal{H}[\pi]]$$

**Why entropy regularization is critical for budget-aware training.** The planner inherits temperature-based stochastic sampling from the underlying VAE architecture, which provides baseline exploration over topologies and roles. However, the introduction of BCFM creates a new failure mode: the budget signal can dominate the modulated embedding, causing the planner to collapse to a single "safe" topology (e.g., always IO under tight budgets) regardless of task content. Explicit entropy regularization counteracts this by maintaining diversity across the budget-conditioned action distribution:

- **Planner**: $\alpha_H^{\text{plan}} = 0.05$. Ensures the budget-conditioned planner explores across topologies rather than collapsing to the lowest-latency-floor option.
- **Executor**: $\alpha_H^{\text{exec}} = 0.02$. The executor's infrastructure-aware state space is high-dimensional ($M(|\Sigma|+2)$ system features). Without entropy, it fixates on whichever model currently has the shortest queue, ignoring quality-latency tradeoffs.

Entropy should be **annealed** over training: start high to explore the budget-topology landscape, reduce as the policy learns meaningful budget-conditioned preferences.

### 8.2 Planner Policy Gradient

The planner uses REINFORCE with an **exponential moving average (EMA) baseline** for variance reduction:

$$\nabla_{\theta_{\text{plan}}} J = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta_{\text{plan}}} \log \pi_{\text{plan}}(a_{\text{plan}}^{(i)} | s_{\text{plan}}^{(i)}) \cdot \hat{A}_{\text{plan}}^{(i)}$$

where the advantage estimate uses an EMA baseline $\bar{G}$:

$$\hat{A}_{\text{plan}}^{(i)} = \tilde{R}_{\text{plan}}^{(i)} - \bar{G}_{\text{plan}}$$

$$\bar{G}_{\text{plan}} \leftarrow (1 - \rho) \cdot \bar{G}_{\text{plan}} + \rho \cdot \text{mean}(\tilde{R}_{\text{plan}})$$

with $\rho = 0.05$ (slow-moving baseline that spans multiple batches).

**Why EMA and not batch mean?** A batch-mean baseline subtracts the current batch average, producing advantage estimates centered at zero for each batch independently. When the policy is near-converged (e.g., all episodes use Chain topology), all rewards are similar and advantages are near-zero noise. An EMA baseline retains memory of past reward levels, producing meaningful advantages when the current batch's rewards differ from the historical average. This provides learning signal even during periods of low diversity.

**Full planner loss:**

$$\mathcal{L}_{\text{plan}} = -\frac{1}{N}\sum_{i} \log \pi_{\text{plan}}(a^{(i)}|s^{(i)}) \cdot \hat{A}^{(i)} - \alpha_H^{\text{plan}} \cdot \mathcal{H}[\pi_{\text{plan}}]$$

### 8.3 Executor Policy Gradient (Actor-Critic)

The executor uses an **actor-critic** architecture with a learned value baseline:

$$\hat{A}_{\text{exec}}^{(k)} = \tilde{R}_{\text{exec}}^{(k)} - V_\phi(s_{\text{exec}}^{(k)})$$

**Full executor loss:**

$$\mathcal{L}_{\text{exec}} = \underbrace{-\frac{1}{K}\sum_k \log \pi_{\text{exec}}(a^{(k)}|s^{(k)}) \cdot \hat{A}^{(k)}}_{\text{actor loss}} + \underbrace{c_V \cdot \frac{1}{K}\sum_k (V_\phi(s^{(k)}) - \tilde{R}^{(k)})^2}_{\text{critic loss}} - \underbrace{\alpha_H^{\text{exec}} \cdot \mathcal{H}[\pi_{\text{exec}}]}_{\text{entropy bonus}}$$

with $c_V = 0.5$.

### 8.4 Mini-Batch Training

Rather than performing a single gradient update per epoch (which wastes the collected experience), transitions are shuffled and split into mini-batches:

- **Mini-batch size**: 32-64 transitions
- **Gradient updates per epoch**: $\lceil N / B \rceil$ where $N$ is the number of collected transitions and $B$ is the batch size
- **Gradient clipping**: $\|\nabla\|_2 \leq 1.0$ for both planner and executor

This ensures the policy receives sufficient gradient updates to learn from the collected experience, especially during early training when episodes are diverse.

### 8.5 Complete Training Algorithm

```
Phase 0 — Transfer Learning Initialization:
  Load pretrained MAS planner checkpoint (task_classifier, collab_determiner,
    num_determiner, role_allocation weights)
  Initialize BCFM and executor networks randomly
  Initialize: lambda_tilde = 0.0, G_bar_plan = 0.0

Phase 1 — End-to-End CMDP Training:
For each sweep config (arrival_rate, pattern, budget):
  For each epoch:
    Collect N episodes (concurrent, Poisson arrivals):
      For each episode i:
        1. Encode query: e_q = Encoder(q_i)
        2. PLANNER: e_tilde = BCFM(e_q, beta_i)    ← budget modulation
           Sample (topology, roles, K) ~ pi_plan(.|e_tilde)
           Store: detached e_q, beta_i, chosen action indices
        3. EXECUTOR: For each role r_k:
           Build action-conditional latency map z_sys
           s_exec = [e_q || e_r_k || b_rem || z_sys]
           Sample (model, strategy) ~ pi_exec(.|s_exec)
           Execute LLM call with EDF priority
        4. Evaluate quality R_quality, observe C_workflow

    Train planner (mini-batch SGD, fresh forward passes):
      Shuffle stored transitions into mini-batches of size B
      For each mini-batch:
        Re-compute: pi_plan = evaluate_plan(e_q, beta, actions)
          ← fresh graph through BCFM + VAE pipeline with current weights
        Compute advantage: A = (R_quality - lambda * C/beta) - G_bar
        L_plan = -mean(log_prob * A) + 0.001 * L_VAE - alpha_H * H[pi]
        Backprop with gradient clipping (max_norm=1.0)
      Update EMA baseline: G_bar = (1-rho)*G_bar + rho*mean(R)

    Train executor (mini-batch actor-critic):
      For each mini-batch of detached executor transitions:
        A = (R_quality - lambda * L_k/b_rem) - V(s_exec)
        L_exec = actor_loss + 0.5*value_loss - alpha_H*entropy
        Backprop with gradient clipping (max_norm=1.0)

    Update Lagrange multiplier (SIGNED dual ascent):
      gap = mean(C_workflow) / mean(beta) - 1.0
      lambda_tilde += eta_lambda * gap    ← gap is signed!
      lambda = softplus(lambda_tilde)
```

**Key training details:**

1. **Transfer learning initialization** (Phase 0): The planner's VAE modules start from pretrained MAS weights, providing task-routing knowledge. BCFM starts near-identity ($\gamma \approx 1$), so the planner initially behaves identically to the pretrained MAS planner. End-to-end fine-tuning then adapts the combined system from MAS's cost-per-token reward to InfraMind's latency-budget reward.

2. **Fresh forward pass at training time**: Planner transitions store only detached inputs and chosen action indices. At training time, the planner re-evaluates these actions via a fresh forward pass through BCFM and the VAE pipeline using current weights. This is the standard RL "evaluate" pattern (as in PPO) and prevents stale computation graphs across optimizer steps.

3. **Concurrent episode collection**: Episodes are collected under Poisson-distributed concurrent load, so the executor encounters realistic infrastructure contention (varying queue depths, cache pressure) during data collection — not simulated load.

---

## 9. Workflow Latency Computation

### 9.1 Topological Execution

The workflow executes in waves based on the topology's DAG structure:

$$\text{wave}_1 = \{r : \text{in\_degree}(r) = 0\}$$
$$\text{wave}_{w+1} = \{r : \text{all predecessors in } \bigcup_{j \leq w} \text{wave}_j\}$$

### 9.2 Critical Path Latency

Since nodes within a wave execute in parallel, the workflow latency is:

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} L_k$$

---

## 10. System Metrics Collection

### 10.1 Real-Time Metrics

The router continuously monitors each LLM server to capture system load:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Running Requests | $n_m^{\text{run}}$ | Active inference requests (prefill + decode) |
| Waiting Requests | $n_m^{\text{wait}}$ | Queued requests awaiting processing |

These metrics are collected via the vLLM `/metrics` endpoint at regular intervals.

---

## 11. Infrastructure-Level EDF Scheduling

### 11.1 Deadline Computation

Each query $q_i$ arrives at time $t_i^{\text{arr}}$ and is assigned a latency budget $\beta_i$. The **deadline** is:

$$D_i = t_i^{\text{arr}} + \beta_i$$

All LLM calls within the same multi-agent workflow inherit the same deadline $D_i$.

### 11.2 Priority Mapping

vLLM's priority scheduler uses a min-heap where lower values are served first:

$$\text{priority}(q_i) = \lfloor D_i \rfloor$$

### 11.3 Interaction with the CMDP

The EDF priority creates a two-level scheduling hierarchy:

| Level | Decision Maker | Mechanism | Controls |
|-------|---------------|-----------|----------|
| **Routing** | CMDP Executor | Policy $\pi_{\text{exec}}$ | *Which* model and strategy to use |
| **Serving** | vLLM Scheduler | EDF priority queue | *When* a request is processed |

---

## 12. InfraMind's Contributions Over Existing MAS Routing

InfraMind builds upon the MAS routing paradigm and introduces five key innovations that transform infrastructure-oblivious task routing into infrastructure-aware, budget-constrained orchestration:

### 12.1 Budget-Conditioned Feature Modulation (BCFM)

**Problem:** Existing MAS planners select collaboration structures based solely on task semantics. They cannot distinguish between a 10-second budget and a 120-second budget for the same query, leading to structurally infeasible plans.

**Our solution:** BCFM is a FiLM conditioning layer ($\tilde{\mathbf{e}}_q = \gamma(\beta) \odot \mathbf{e}_q + \beta(\beta)$) inserted before the planner pipeline. It modulates the query embedding based on the latency budget, steering structural decisions without modifying the planner's internal architecture. This design enables direct transfer learning from pretrained MAS checkpoints — the planner modules receive the same-dimensional input, but the input distribution is now budget-conditioned.

### 12.2 Hierarchical CMDP with Structural-Resource Separation

**Problem:** Jointly optimizing topology, roles, models, and strategies in a single policy creates a combinatorial action space that is intractable to explore.

**Our solution:** A two-level hierarchy that separates concerns:
- **Planner** (structural): Selects topology + roles + agent count based on query + budget. Operates once at $t=0$.
- **Executor** (resource): Selects model + strategy per agent based on query + role + remaining budget + real-time system state. Operates at each node.

Each level has its own policy, optimizer, and reward signal, reducing the effective action space from $|\mathcal{T}| \times |\mathcal{R}|^K \times |\mathcal{M}|^K \times |\Sigma|^K$ to $|\mathcal{T}| \times |\mathcal{R}|^K$ (planner) + $|\mathcal{M}| \times |\Sigma|$ per node (executor).

### 12.3 Infrastructure-Aware Executor with Action-Conditional Latency Maps

**Problem:** Existing MAS routers select models based on task-model affinity alone, ignoring real-time infrastructure state. Under concurrent load, this creates queue imbalance: preferred models accumulate deep queues while others sit idle.

**Our solution:** The executor observes an **action-conditional latency map** — for every candidate (model, strategy) pair, it sees the predicted end-to-end latency under current infrastructure conditions:

$$\mathbf{z}_m = [\hat{L}_{m,\sigma_1} \| \cdots \| \hat{L}_{m,\sigma_{|\Sigma|}} \| n_m^{\text{run}} \| n_m^{\text{wait}}]$$

This gives the executor a complete "price list" of all possible actions, enabling it to trade off quality vs. latency in real time. Under high load, it routes to underutilized models; under low load, it invests in richer reasoning.

### 12.4 Lagrangian Relaxation with Signed Dual Ascent

**Problem:** Hard budget constraints produce zero gradient when satisfied, removing the incentive to differentiate between fast and slow actions that both meet the deadline.

**Our solution:** Proportional latency cost ($C/\beta$, always active) combined with signed dual ascent on the Lagrange multiplier $\lambda$. The signed update is critical: when average latency is below budget, $\lambda$ **decreases**, relaxing latency pressure and allowing the policy to explore higher-quality but more expensive configurations. This produces an equilibrium where $\lambda$ oscillates around the constraint boundary rather than growing monotonically toward collapse.

### 12.5 Deadline-Aware Serving via EDF Scheduling

**Problem:** Standard FCFS scheduling at the serving layer is deadline-blind — a request with 2 seconds remaining waits behind a request with 60 seconds remaining, causing avoidable budget violations.

**Our solution:** Each query's deadline ($D_i = t_i^{\text{arr}} + \beta_i$) is propagated to the vLLM serving layer as an EDF priority. Requests closer to their deadline are served first, reducing tail-latency violations at the infrastructure level. This creates a cooperative two-level scheduling hierarchy: the CMDP executor decides *which* model to use, and EDF scheduling decides *when* the request is processed.

### 12.6 Summary of Innovations

| Innovation | What It Addresses | Level |
|-----------|-------------------|-------|
| **BCFM** | Budget-blind structural planning | Planner input |
| **Hierarchical CMDP** | Intractable joint action space | Architecture |
| **Action-conditional latency map** | Infrastructure-oblivious model selection | Executor state |
| **Signed dual ascent** | Lagrangian collapse / monotonic $\lambda$ growth | Training |
| **EDF scheduling** | Deadline-blind serving | Infrastructure |

---

## 13. Summary of Key Equations

| Component | Equation |
|-----------|----------|
| **Action-Conditional Latency** | $\hat{L}_{m,\sigma} = \hat{T}_{m,\sigma}^{\text{TTFT}} + \hat{T}_{m,\sigma}^{\text{TPOT}} \cdot \hat{N}_{m,\sigma}^{\text{out}}$ |
| **BCFM (Planner State)** | $\tilde{\mathbf{e}}_q = \gamma(\beta_i) \odot \mathbf{e}_q + \beta(\beta_i)$ |
| **Planner CMDP Reward** | $\tilde{R}_{\text{plan}} = R_{\text{quality}} - \lambda \cdot C_{\text{workflow}} / \beta$ |
| **Executor State** | $s_{\text{exec}} = [\mathbf{e}_q \| \mathbf{e}_r \| b_{\text{rem}} \| \mathbf{z}_{\text{sys}}]$ |
| **Executor CMDP Reward** | $\tilde{R}_{\text{exec}} = R_{\text{quality}} - \lambda \cdot L_k / b_{\text{rem}}$ |
| **System State (per model)** | $\mathbf{z}_m = [\hat{L}_{m,\sigma_1} \| \cdots \| \hat{L}_{m,\sigma_{|\Sigma|}} \| n_m^{\text{run}} \| n_m^{\text{wait}}]$ |
| **Lagrangian Reward** | $\tilde{R} = R_{\text{quality}} - \lambda \cdot C / \beta$ |
| **Constraint** | $C_{\text{workflow}} \leq \beta$ |
| **Multiplier Update (signed)** | $\lambda \leftarrow \max(0, \lambda + \eta_\lambda (\bar{C}/\bar{\beta} - 1))$ |
| **Policy Gradient** | $\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot \hat{A}] + \alpha_H \nabla \mathcal{H}[\pi]$ |
| **EMA Baseline** | $\bar{G} \leftarrow (1-\rho)\bar{G} + \rho \cdot \text{mean}(G)$ |
| **Deadline (EDF)** | $D_i = t_i^{\text{arr}} + \beta_i$ |
| **EDF Priority** | $\text{priority}(q_i) = \lfloor D_i \rfloor$ |

---

## 14. Notation Reference

| Symbol | Description |
|--------|-------------|
| $q$ | Input query |
| $\mathbf{e}_q$ | Query embedding |
| $\tilde{\mathbf{e}}_q$ | Budget-conditioned query embedding (BCFM output) |
| $\mathcal{M}$ | Set of available LLMs |
| $\mathcal{T}$ | Set of topologies |
| $\mathcal{R}$ | Set of roles |
| $\Sigma$ | Set of prompting strategies |
| $\tau$ | Selected topology |
| $m$ | Selected LLM |
| $\sigma$ | Selected strategy |
| $\gamma(\beta), \beta(\beta)$ | BCFM scale and shift parameters (FiLM) |
| $\beta$ | Latency budget |
| $b_{\text{rem}}$ | Remaining budget |
| $\lambda$ | Lagrange multiplier |
| $\tilde{\lambda}$ | Raw Lagrange parameter (before softplus) |
| $\alpha_H$ | Entropy regularization coefficient |
| $\rho$ | EMA baseline smoothing factor |
| $L$ | Observed latency |
| $\hat{L}_{m,\sigma}$ | Action-conditional predicted latency |
| $n^{\text{run}}$ | Running requests |
| $n^{\text{wait}}$ | Waiting requests |
| $\pi_{\text{plan}}$ | Planner policy |
| $\pi_{\text{exec}}$ | Executor policy |
| $V(s)$ | Learned value function (executor baseline) |
| $\bar{G}$ | EMA reward baseline (planner) |
| $D_i$ | Deadline of query $q_i$ |

---

*This document serves as a theoretical reference for the InfraMind implementation.*
