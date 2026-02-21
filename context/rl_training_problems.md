# InfraMind: Help Design the Reward Function

## What is InfraMind?

InfraMind is an infrastructure-aware multi-agent LLM orchestration system with two-level RL:

- **Planner** (REINFORCE): Picks collaboration topology (IO/CoT/Chain/Debate/Reflection/FullConnected) and assigns roles. Sees only the query embedding — no budget awareness.
- **Executor** (Actor-Critic): For each role in the workflow, picks which LLM model and which prompting strategy. Sees query embedding + role embedding + remaining budget + predicted latency map for all 15 (model, strategy) pairs.

### Model Pool (5 models)
| # | Model | Capability | Latency |
|---|-------|-----------|---------|
| 1 | DeepSeek-R1-Distill-Qwen-32B | Strongest | ~3-8s |
| 2 | Mistral-Small-24B-Instruct | Strong | ~2-5s |
| 3 | Qwen2.5-Coder-14B-Instruct | Medium | ~1-4s |
| 4 | Llama-3.1-8B-Instruct | Weak | ~0.5-3s |
| 5 | Llama-3.2-3B-Instruct | Weakest | ~0.5-2s |

### Prompt Strategies (3 strategies)
- **Flash**: "Go straight to the answer." (fastest, lowest quality)
- **Concise**: "Brief 2-3 key points, then answer." (balanced)
- **DeepThink**: "Reason thoroughly step-by-step." (slowest, highest quality)

### Quality Signal
- **Binary `is_solved` ∈ {0, 1}**: Exact match for math. This is the ONLY ground truth — no partial credit.
- We also have a **quality predictor** (trained MLP) that predicts LLM-as-a-Judge scores in [0, 10] given (query, response, model, role, strategy). Not ground truth, just an estimate.

### Budget
- Each query gets a latency budget sampled from `LogUniform(5, 300)` seconds.
- Actual workflow latencies are typically 2-8 seconds.
- Budget is given to the executor as part of its state (budget_remaining).
- The planner does NOT see the budget — only the executor does.

### Training Setup
- 519 MATH training items, 131 validation items.
- Each epoch = 7 arrival rate sweeps (10, 30, 50, 100, 150, 200, 300 req/min).
- Batch size 64. One trajectory per query per sweep.
- Planner: REINFORCE with batch-mean baseline subtraction.
- Executor: Actor-Critic (A2C) with normalized advantages (zero mean, unit variance).
- Entropy coefficient: 0.10. Value coefficient: 0.5. Grad clip: 1.0.

---

## What Happened During Training

### Old Reward (the one that failed)

```
Planner:  utility = is_solved - max(0, workflow_latency / budget - 1.0)
Executor: reward  = is_solved - max(0, workflow_latency / budget - 1.0)
```

Wrong answer = 0.0 reward. Correct answer within budget = 1.0. Correct answer over budget = negative (can go to -4.0 or worse).

**Critical flaw**: Wrong+fast (0.0) scored BETTER than correct+over-budget (-1.0). The policy learned: "be wrong and fast."

### What collapsed

Over 30 epochs:
- **Accuracy flatlined at ~55%** (validation best: 55.0% at epoch 2, then regressed)
- **Model**: DeepSeek-32B dropped 16.5% → 1.9%. Llama-3B rose 22.9% → 41.6%.
- **Strategy**: DeepThink dropped 23.1% → 8.9%. Flash+Concise dominated 91%.
- **Topology**: IO (single agent) dominated 60-67%. Multi-agent topologies abandoned.
- **Latency dropped from 8s → 2s** but quality didn't improve at all.

### Baseline comparison (same 519 items, no budget, no strategies)

The baseline MAS router (no InfraMind, no budget awareness, no prompt strategies) on the same data:
- **Achieved 73.2% accuracy** (vs InfraMind's 55%)
- Also converged to small models (Llama-3B 37.8%, Llama-8B 40.6%) — similar to InfraMind
- But it uses the full prompt always (no Flash/Concise/DeepThink degradation)
- Topology: IO 39.5%, CoT 42.6% — similar pattern

So the baseline also favors cheap models but gets 73% because it doesn't degrade prompts.

---

## Our Current Reward (implemented but not yet tested)

```python
# Both planner and executor use the same formula:
if is_solved:
    penalty = min(0.5, max(0, workflow_latency / budget - 1.0) * 0.3)
    reward = 1.0 - penalty    # range [0.50, 1.0]
else:
    reward = -1.0             # always negative
```

| Scenario | Old Reward | New Reward |
|----------|-----------|------------|
| Correct + within budget | +1.0 | +1.0 |
| Correct + 2x over budget | -1.0 | +0.70 |
| Correct + 5x over budget | -4.0 | +0.50 |
| Wrong + fast | 0.0 | **-1.0** |

Correct ALWAYS outranks wrong (min gap = 1.50).

---

## Our Goal

We want InfraMind to **dominate the baseline on the quality-latency Pareto curve**:

1. **Quality is the FIRST priority.** A wrong answer is worthless regardless of latency.
2. **When budget is generous, use it.** Invest in 32B models, DeepThink strategy, multi-agent topologies to maximize solve rate. Don't leave budget on the table.
3. **When budget is tight, gracefully degrade.** Use cheaper configs but still attempt correctness.
4. **SLO compliance.** Meet budget when possible, but never sacrifice correctness to be fast.

---

## Remaining Concerns

1. **No incentive to USE generous budget.** If the answer is correct and within budget, reward is always +1.0 whether latency was 2s or 55s on a 60s budget. The executor has no reason to pick 32B+DeepThink over 3B+Flash when both are within budget — unless the quality difference shows up in is_solved statistics over many samples.

2. **Binary reward = high variance.** If 3B+Flash solves 40% and 32B+DeepThink solves 65%, the policy needs many samples to learn this 25% difference from binary {0,1} outcomes. With 519 items and one trajectory per query, this is noisy.

3. **Episode-level credit assignment.** Every executor step in a workflow gets the same reward. A 6-agent debate where one step used DeepThink and five used Flash — all get the same +1 or -1.

4. **Planner advantages are not variance-normalized.** The executor normalizes advantages to (zero mean, unit variance), but the planner only subtracts the mean. This can cause unstable gradients.

5. **Quality predictor is available but underutilized.** It predicts [0, 10] per step. Currently passed to the executor reward function but not used in the formula (the new reward is purely binary). Could it provide a denser signal?

---

## What We Need From You

Given everything above, **propose the best reward function for both planner and executor.** Consider:

- Should we use the quality predictor as a shaping signal? How?
- Should correct-within-budget reward depend on how much budget was used (incentivize using budget for quality)?
- How should we handle credit assignment for multi-step executor decisions?
- Should the planner advantages be normalized differently?
- Any other changes to training dynamics (entropy, learning rate, batch structure)?

Please provide:
- Mathematical formulation for both planner reward and executor reward
- PyTorch-style pseudocode
- Explanation of why each design choice helps

Constraints:
- Quality is binary (is_solved ∈ {0, 1}) — cannot change this
- Must converge within 10-50 epochs (1 hour per epoch)
- PyTorch, simple modifications to reward/advantage computation
- Planner and executor are trained with separate optimizers
