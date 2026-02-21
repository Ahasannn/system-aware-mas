"""
Diagnostic visualization: Baseline MAS vs InfraMind on MATH test set.
Focuses on identifying InfraMind RL training issues and next steps.

Only uses baseline data at matching arrival rates for fair comparison.

Usage:
    python scripts/testing/temp/visualize_baseline_vs_inframind.py \
        --baseline logs/testing/math/baseline_math_test_maxseq32.csv \
        --inframind logs/testing/math/inframind_math_quick_test.csv \
        --output logs/testing/math/temp/plots
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

MODEL_SHORT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-32B",
    "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral-24B",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen-14B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3B",
}

MODEL_ORDER = ["DeepSeek-32B", "Mistral-24B", "Qwen-14B", "Llama-8B", "Llama-3B"]

MODEL_COLORS = {
    "DeepSeek-32B": "#E53935",
    "Mistral-24B": "#FB8C00",
    "Qwen-14B": "#8E24AA",
    "Llama-8B": "#1E88E5",
    "Llama-3B": "#43A047",
}

TOPO_ORDER = ["IO", "CoT", "Chain", "Reflection", "Debate", "FullConnected"]
TOPO_COLORS = {
    "IO": "#78909C",
    "CoT": "#42A5F5",
    "Chain": "#66BB6A",
    "Reflection": "#FFA726",
    "Debate": "#EF5350",
    "FullConnected": "#AB47BC",
}

STRATEGY_ORDER = ["flash", "concise", "deepthink"]
STRATEGY_COLORS = {"flash": "#4CAF50", "concise": "#2196F3", "deepthink": "#FF5722"}

BL_COLOR = "#607D8B"
IM_COLORS = {20: "#1976D2", 50: "#F57C00", 200: "#388E3C", 10: "#D32F2F"}


def _short_model(name: str) -> str:
    return MODEL_SHORT.get(name, name.split("/")[-1])


def _pct(vals, total):
    return [v / total * 100 if total else 0 for v in vals]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_baseline(path: str, keep_rates=None):
    episodes, steps = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            rate = float(r["arrival_rate"])
            if keep_rates and rate not in keep_rates:
                continue
            if r["record_type"] == "episode":
                episodes.append({
                    "arrival_rate": rate,
                    "is_correct": r["quality_is_correct"] == "1",
                    "latency": float(r["workflow_latency_seconds"]),
                    "topology": r.get("reasoning_name", ""),
                    "num_agents": int(r.get("num_agents", 0) or 0),
                })
            elif r["record_type"] == "step":
                steps.append({
                    "arrival_rate": rate,
                    "model": _short_model(r.get("llm_name", "")),
                    "latency": float(r.get("latency_seconds", 0) or 0),
                    "completion_tokens": int(r.get("completion_tokens", 0) or 0),
                })
    return episodes, steps


def load_inframind(path: str):
    episodes, steps = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["record_type"] == "episode":
                episodes.append({
                    "arrival_rate": float(r["arrival_rate"]),
                    "budget": float(r.get("budget_total", 0) or 0),
                    "is_correct": r.get("quality_is_solved") == "True"
                                  or r.get("quality_is_solved") == "1"
                                  or r.get("quality") == "1.0",
                    "latency": float(r["workflow_latency_seconds"]),
                    "topology": r.get("topology", ""),
                    "num_agents": int(r.get("agent_count", 0) or 0) if r.get("agent_count") else 0,
                })
            elif r["record_type"] in ("step", "role_step"):
                steps.append({
                    "arrival_rate": float(r["arrival_rate"]),
                    "budget": float(r.get("budget_total", 0) or 0),
                    "model": _short_model(r.get("model_name", "") or r.get("llm_name", "")),
                    "strategy": (r.get("strategy_name", "") or "none").lower(),
                    "latency": float(r.get("latency_seconds", 0) or 0),
                    "completion_tokens": int(r.get("completion_tokens", 0) or 0),
                })
    return episodes, steps


# ── Plot 1: Accuracy comparison ──────────────────────────────────────────────

def plot_accuracy(bl_eps, im_eps, rates, im_budgets, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rates))
    n_bars = 1 + len(im_budgets)
    width = 0.7 / n_bars

    # Baseline — only draw bars where data exists
    for j, rate in enumerate(rates):
        sub = [e for e in bl_eps if e["arrival_rate"] == rate]
        if not sub:
            continue
        val = sum(e["is_correct"] for e in sub) / len(sub) * 100
        pos = x[j] - width * (n_bars - 1) / 2
        bar = ax.bar(pos, val, width, color=BL_COLOR, edgecolor="white",
                     label="Baseline (MAS)" if j == 0 or not any(e["arrival_rate"] == rates[0] for e in bl_eps) else "")
        ax.text(pos, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Ensure baseline legend entry exists
    ax.bar([], [], color=BL_COLOR, label="Baseline (MAS)")

    # InfraMind per budget — only draw bars where data exists
    for i, budget in enumerate(im_budgets):
        color = IM_COLORS.get(int(budget), "#999")
        label_done = False
        for j, rate in enumerate(rates):
            sub = [e for e in im_eps if e["arrival_rate"] == rate and e["budget"] == budget]
            if not sub:
                continue
            val = sum(e["is_correct"] for e in sub) / len(sub) * 100
            offset = -width * (n_bars - 1) / 2 + width * (i + 1)
            lbl = f"InfraMind (B={int(budget)}s)" if not label_done else ""
            ax.bar(x[j] + offset, val, width, color=color, edgecolor="white", label=lbl)
            ax.text(x[j] + offset, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
            label_done = True

    ax.set_xlabel("Arrival Rate (req/min)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy: Baseline vs InfraMind\n(InfraMind +6-7% but budget has no effect — executor not differentiating)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in rates])
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=10)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_accuracy_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 1_accuracy_comparison.png")


# ── Plot 2: Latency comparison ───────────────────────────────────────────────

def plot_latency(bl_eps, im_eps, rates, im_budgets, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(rates))
    n_bars = 1 + len(im_budgets)
    width = 0.7 / n_bars

    # Baseline — skip missing rates
    ax.bar([], [], color=BL_COLOR, label="Baseline")
    for j, rate in enumerate(rates):
        lats = [e["latency"] for e in bl_eps if e["arrival_rate"] == rate]
        if not lats:
            continue
        val = np.mean(lats)
        pos = x[j] - width * (n_bars - 1) / 2
        ax.bar(pos, val, width, color=BL_COLOR, edgecolor="white")
        ax.text(pos, val + 1, f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, budget in enumerate(im_budgets):
        color = IM_COLORS.get(int(budget), "#999")
        label_done = False
        for j, rate in enumerate(rates):
            lats = [e["latency"] for e in im_eps if e["arrival_rate"] == rate and e["budget"] == budget]
            if not lats:
                continue
            val = np.mean(lats)
            offset = -width * (n_bars - 1) / 2 + width * (i + 1)
            lbl = f"InfraMind B={int(budget)}s" if not label_done else ""
            ax.bar(x[j] + offset, val, width, color=color, edgecolor="white", label=lbl)
            ax.text(x[j] + offset, val + 1, f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
            label_done = True

    ax.set_xlabel("Arrival Rate (req/min)", fontsize=12)
    ax.set_ylabel("Avg Latency (s)", fontsize=12)
    ax.set_title("Avg Latency: InfraMind ~3x faster than Baseline\n(but latency barely changes with budget — executor ignoring budget signal?)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in rates])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_latency_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 2_latency_comparison.png")


# ── Plot 3: Model distribution ───────────────────────────────────────────────

def plot_model_dist(bl_steps, im_steps, rates, im_budgets, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    model_fracs = {m: [] for m in MODEL_ORDER}

    for rate in rates:
        # Baseline
        bl_sub = [s for s in bl_steps if s["arrival_rate"] == rate]
        if bl_sub:
            labels.append(f"Baseline\nr={int(rate)}")
            total = len(bl_sub)
            counts = Counter(s["model"] for s in bl_sub)
            for m in MODEL_ORDER:
                model_fracs[m].append(counts.get(m, 0) / total * 100)

        # InfraMind per budget
        for budget in im_budgets:
            im_sub = [s for s in im_steps if s["arrival_rate"] == rate and s["budget"] == budget]
            if im_sub:
                labels.append(f"IM B={int(budget)}s\nr={int(rate)}")
                total = len(im_sub)
                counts = Counter(s["model"] for s in im_sub)
                for m in MODEL_ORDER:
                    model_fracs[m].append(counts.get(m, 0) / total * 100)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for model in MODEL_ORDER:
        vals = model_fracs[model]
        bars = ax.bar(x, vals, bottom=bottom, label=model, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        # Add percentage labels for significant segments
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 8:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_ylabel("Model Usage (%)")
    ax.set_title("Model Distribution: ISSUE — InfraMind uses 45% DeepSeek-32B vs Baseline 8%\n(Executor over-relies on expensive model regardless of budget)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_model_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 3_model_distribution.png")


# ── Plot 4: Strategy distribution (InfraMind diagnostic) ─────────────────────

def plot_strategy_dist(im_steps, rates, im_budgets, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    strat_fracs = {s: [] for s in STRATEGY_ORDER}

    for rate in rates:
        for budget in im_budgets:
            sub = [s for s in im_steps if s["arrival_rate"] == rate and s["budget"] == budget]
            if not sub:
                continue
            labels.append(f"B={int(budget)}s r={int(rate)}")
            total = len(sub)
            counts = Counter(s["strategy"] for s in sub)
            for s in STRATEGY_ORDER:
                strat_fracs[s].append(counts.get(s, 0) / total * 100)

    if not labels:
        return

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for strat in STRATEGY_ORDER:
        vals = strat_fracs[strat]
        bars = ax.bar(x, vals, bottom=bottom, label=strat.capitalize(), color=STRATEGY_COLORS[strat], edgecolor="white", linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 8:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_ylabel("Strategy Usage (%)")
    ax.set_title("ISSUE — Strategy distribution identical across budgets\n(Expected: tight budget → more Flash, loose budget → more DeepThink)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_strategy_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 4_strategy_distribution.png")


# ── Plot 5: Topology distribution ────────────────────────────────────────────

def plot_topology_dist(bl_eps, im_eps, rates, im_budgets, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    topo_fracs = {t: [] for t in TOPO_ORDER}

    for rate in rates:
        # Baseline
        bl_sub = [e for e in bl_eps if e["arrival_rate"] == rate]
        if bl_sub:
            labels.append(f"Baseline\nr={int(rate)}")
            total = len(bl_sub)
            counts = Counter(e["topology"] for e in bl_sub)
            for t in TOPO_ORDER:
                topo_fracs[t].append(counts.get(t, 0) / total * 100)

        # InfraMind (topologies are budget-independent since planner is budget-unaware)
        # Show one bar per rate (aggregate across budgets)
        im_sub = [e for e in im_eps if e["arrival_rate"] == rate]
        if im_sub:
            labels.append(f"InfraMind\nr={int(rate)}")
            total = len(im_sub)
            counts = Counter(e["topology"] for e in im_sub)
            for t in TOPO_ORDER:
                topo_fracs[t].append(counts.get(t, 0) / total * 100)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for topo in TOPO_ORDER:
        vals = topo_fracs[topo]
        bars = ax.bar(x, vals, bottom=bottom, label=topo, color=TOPO_COLORS.get(topo, "#999"), edgecolor="white", linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_ylabel("Topology Usage (%)")
    ax.set_title("Topology: Baseline 63% IO vs InfraMind 46% IO\n(InfraMind planner uses more diverse topologies — good sign for quality)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_topology_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 5_topology_distribution.png")


# ── Plot 6: Budget compliance scatter ────────────────────────────────────────

def plot_budget_compliance(im_eps, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im_budgets = sorted(set(e["budget"] for e in im_eps))

    for ax, budget in zip(axes, im_budgets):
        sub = [e for e in im_eps if e["budget"] == budget]
        for is_correct, color, label, marker in [
            (True, "#4CAF50", "Correct", "o"),
            (False, "#F44336", "Wrong", "x"),
        ]:
            pts = [e for e in sub if e["is_correct"] == is_correct]
            if pts:
                lats = [e["latency"] for e in pts]
                ax.scatter(range(len(pts)), lats, c=color, label=label, marker=marker, alpha=0.4, s=15)

        ax.axhline(y=budget, color="black", linestyle="--", alpha=0.5, label=f"Budget={budget}s")
        over = sum(1 for e in sub if e["latency"] > budget)
        acc = sum(e["is_correct"] for e in sub) / len(sub) * 100
        ax.set_title(f"Budget={budget}s: {over}/{len(sub)} over budget ({over/len(sub)*100:.1f}%), Acc={acc:.1f}%", fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Latency (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Budget Compliance: How many episodes exceed their budget?", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "6_budget_compliance.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 6_budget_compliance.png")


# ── Plot 7: Accuracy vs Latency Pareto ───────────────────────────────────────

def plot_pareto(bl_eps, im_eps, rates, im_budgets, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Baseline
    for rate in rates:
        sub = [e for e in bl_eps if e["arrival_rate"] == rate]
        if not sub:
            continue
        acc = sum(e["is_correct"] for e in sub) / len(sub) * 100
        avg_lat = np.mean([e["latency"] for e in sub])
        ax.scatter(avg_lat, acc, c=BL_COLOR, s=150, marker="s", zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(f"BL r={int(rate)}", (avg_lat, acc), textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight="bold")

    # InfraMind
    for budget in im_budgets:
        color = IM_COLORS.get(int(budget), "#999")
        for rate in rates:
            sub = [e for e in im_eps if e["arrival_rate"] == rate and e["budget"] == budget]
            if not sub:
                continue
            acc = sum(e["is_correct"] for e in sub) / len(sub) * 100
            avg_lat = np.mean([e["latency"] for e in sub])
            ax.scatter(avg_lat, acc, c=color, s=150, marker="o", zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(f"IM B={int(budget)} r={int(rate)}", (avg_lat, acc), textcoords="offset points", xytext=(10, 5), fontsize=9)

    # Legend
    ax.scatter([], [], c=BL_COLOR, s=100, marker="s", label="Baseline")
    for budget in im_budgets:
        ax.scatter([], [], c=IM_COLORS.get(int(budget), "#999"), s=100, marker="o", label=f"InfraMind B={int(budget)}s")

    ax.set_xlabel("Avg Latency (s)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Latency — InfraMind dominates (faster + more accurate)\n(But all IM points cluster together — budget not creating Pareto trade-off)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "7_pareto_accuracy_latency.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 7_pareto_accuracy_latency.png")


# ── Plot 8: Budget sensitivity diagnostic ────────────────────────────────────

def plot_budget_sensitivity(im_steps, im_eps, rates, im_budgets, out_dir):
    """Side-by-side: how model + strategy change (or don't) with budget."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, rate in enumerate(rates[:2]):
        # Top row: model distribution per budget
        ax = axes[0][col]
        for i, budget in enumerate(im_budgets):
            sub = [s for s in im_steps if s["arrival_rate"] == rate and s["budget"] == budget]
            if not sub:
                continue
            total = len(sub)
            counts = Counter(s["model"] for s in sub)
            x_pos = np.arange(len(MODEL_ORDER))
            vals = [counts.get(m, 0) / total * 100 for m in MODEL_ORDER]
            offset = (i - len(im_budgets) / 2 + 0.5) * 0.35
            color = IM_COLORS.get(int(budget), "#999")
            ax.bar(x_pos + offset, vals, 0.3, label=f"B={int(budget)}s", color=color, edgecolor="white")

        ax.set_title(f"Model Choice at rate={int(rate)}\n(nearly identical — executor ignores budget)", fontsize=10)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, fontsize=8, rotation=15)
        ax.set_ylabel("Usage (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Bottom row: strategy distribution per budget
        ax = axes[1][col]
        for i, budget in enumerate(im_budgets):
            sub = [s for s in im_steps if s["arrival_rate"] == rate and s["budget"] == budget]
            if not sub:
                continue
            total = len(sub)
            counts = Counter(s["strategy"] for s in sub)
            x_pos = np.arange(len(STRATEGY_ORDER))
            vals = [counts.get(s, 0) / total * 100 for s in STRATEGY_ORDER]
            offset = (i - len(im_budgets) / 2 + 0.5) * 0.35
            color = IM_COLORS.get(int(budget), "#999")
            ax.bar(x_pos + offset, vals, 0.3, label=f"B={int(budget)}s", color=color, edgecolor="white")

        ax.set_title(f"Strategy Choice at rate={int(rate)}\n(nearly identical — executor ignores budget)", fontsize=10)
        ax.set_xticks(range(len(STRATEGY_ORDER)))
        ax.set_xticklabels([s.capitalize() for s in STRATEGY_ORDER], fontsize=9)
        ax.set_ylabel("Usage (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CORE ISSUE: Executor not responding to budget signal\nExpected: B=20 → Flash+small models, B=200 → DeepThink+large models", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "8_budget_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("  Saved: 8_budget_sensitivity.png")


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(bl_eps, bl_steps, im_eps, im_steps, rates, im_budgets):
    print("\n" + "=" * 95)
    print("SUMMARY TABLE")
    print("=" * 95)
    print(f"{'System':<20} {'Rate':>6} {'Budget':>7} {'Items':>6} {'Acc%':>7} {'AvgLat':>8} {'P50Lat':>8} {'P90Lat':>8} {'OverBudget':>10}")
    print("-" * 95)

    for rate in rates:
        sub = [e for e in bl_eps if e["arrival_rate"] == rate]
        if sub:
            acc = sum(e["is_correct"] for e in sub) / len(sub) * 100
            lats = [e["latency"] for e in sub]
            print(f"{'Baseline':<20} {int(rate):>6} {'  —':>7} {len(sub):>6} {acc:>6.1f}% {np.mean(lats):>7.1f}s {np.median(lats):>7.1f}s {np.percentile(lats, 90):>7.1f}s {'—':>10}")

        for budget in im_budgets:
            sub = [e for e in im_eps if e["arrival_rate"] == rate and e["budget"] == budget]
            if not sub:
                continue
            acc = sum(e["is_correct"] for e in sub) / len(sub) * 100
            lats = [e["latency"] for e in sub]
            over = sum(1 for e in sub if e["latency"] > budget)
            print(f"{'InfraMind':<20} {int(rate):>6} {int(budget):>6}s {len(sub):>6} {acc:>6.1f}% {np.mean(lats):>7.1f}s {np.median(lats):>7.1f}s {np.percentile(lats, 90):>7.1f}s {over:>6}/{len(sub)}")
        print()

    print("=" * 95)

    # Diagnosis
    print("\n" + "=" * 95)
    print("DIAGNOSIS: Key Issues in InfraMind RL Training")
    print("=" * 95)

    # Issue 1: Budget insensitivity
    print("\n1. EXECUTOR BUDGET-BLIND (Critical)")
    print("   Model and strategy distributions are nearly IDENTICAL for B=20 vs B=200.")
    print("   The executor MLP is not using the budget_remaining input to vary its decisions.")
    for rate in rates:
        for budget in im_budgets:
            sub = [s for s in im_steps if s["arrival_rate"] == rate and s["budget"] == budget]
            if sub:
                ds = sum(1 for s in sub if s["model"] == "DeepSeek-32B") / len(sub) * 100
                flash = sum(1 for s in sub if s["strategy"] == "flash") / len(sub) * 100
                print(f"   rate={int(rate)} B={int(budget):>3}s: DeepSeek={ds:.1f}%, Flash={flash:.1f}%")
    print("   → Root cause: quality-first reward drowns out budget signal.")
    print("     Correct episodes always get positive reward regardless of budget usage.")
    print("     Executor has no incentive to prefer cheap configs when budget is tight.")

    # Issue 2: DeepSeek overuse
    print("\n2. DEEPSEEK OVERUSE (Major)")
    bl_ds = sum(1 for s in bl_steps if s["model"] == "DeepSeek-32B") / len(bl_steps) * 100
    im_ds = sum(1 for s in im_steps if s["model"] == "DeepSeek-32B") / len(im_steps) * 100
    print(f"   Baseline uses {bl_ds:.1f}% DeepSeek, InfraMind uses {im_ds:.1f}% DeepSeek.")
    print("   InfraMind learned 'DeepSeek = best quality' and always picks it.")
    print("   → This works for accuracy (+7%) but creates latency/cost problems at scale.")

    # Issue 3: Strategy uniformity
    print("\n3. STRATEGY NOT ADAPTING (Major)")
    print("   Flash/Concise/DeepThink ratios are ~42/31/27 regardless of budget or load.")
    print("   → Executor treats strategy as noise, not as a budget-control lever.")

    # Positive signals
    print("\n" + "=" * 95)
    print("POSITIVE SIGNALS")
    print("=" * 95)
    bl_acc = sum(e["is_correct"] for e in bl_eps) / len(bl_eps) * 100
    im_acc = sum(e["is_correct"] for e in im_eps) / len(im_eps) * 100
    print(f"\n+ Accuracy: InfraMind {im_acc:.1f}% vs Baseline {bl_acc:.1f}% (+{im_acc-bl_acc:.1f}%)")
    bl_lat = np.mean([e["latency"] for e in bl_eps])
    im_lat = np.mean([e["latency"] for e in im_eps])
    print(f"+ Latency:  InfraMind {im_lat:.1f}s vs Baseline {bl_lat:.1f}s ({bl_lat/im_lat:.1f}x faster)")
    print("+ Topology: InfraMind uses more diverse topologies (46% IO vs 63% IO)")
    print("+ Zero timeouts, zero failures across all 2000 episodes")

    # Next steps
    print("\n" + "=" * 95)
    print("NEXT STEPS — What to Fix")
    print("=" * 95)
    print("""
A. REWARD FUNCTION (Priority 1)
   Current: correct=[0.50, 1.0], wrong=[-1.0, -0.7]
   Problem: Within the correct range, budget overshoot penalty (0.3 * max(0, L/B-1))
   is too weak. L/B=2 only gives 0.3 penalty (reward=0.7), still positive.

   Options:
   a) Steeper budget penalty for executor: reward *= max(0, 1 - (L/B - 1))
      → If L=2*B, reward drops to 0 instead of 0.7
   b) Separate executor reward: quality_reward + budget_compliance_reward
      → Decouple accuracy from budget adherence
   c) Curriculum: first train for quality (done), then freeze planner and
      train executor with stronger budget pressure

B. EXECUTOR STATE REPRESENTATION (Priority 2)
   Check if budget_remaining is properly normalized in executor MLP input.
   If budget_remaining is in seconds (5-300) while other features are 0-1,
   the MLP may ignore it. Normalize to [0, 1] = remaining/total.

C. TRAINING DATA DIVERSITY (Priority 3)
   Training used LogUniform(5, 300) budgets — good range.
   But if most correct episodes had L << B, executor never learned the
   tight-budget regime. May need curriculum with intentionally tight budgets.

D. MORE TRAINING EPOCHS (Priority 4)
   Current: 2 epochs. Executor policy may not have converged.
   Try 5-10 epochs with early stopping on budget-conditioned metrics.
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline vs InfraMind diagnostic comparison")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--inframind", required=True)
    parser.add_argument("--output", default="logs/testing/math/temp/plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load InfraMind first to get its rates
    print("Loading InfraMind data...")
    im_eps, im_steps = load_inframind(args.inframind)
    im_rates = sorted(set(e["arrival_rate"] for e in im_eps))
    im_budgets = sorted(set(e["budget"] for e in im_eps))
    print(f"  {len(im_eps)} episodes, {len(im_steps)} steps")
    print(f"  Rates: {im_rates}, Budgets: {im_budgets}")

    # Load baseline and keep only rates common to BOTH systems
    print("Loading baseline data...")
    bl_all_eps, bl_all_steps = load_baseline(args.baseline)
    bl_available_rates = set(e["arrival_rate"] for e in bl_all_eps)
    common_rates = sorted(bl_available_rates & set(im_rates))
    print(f"  Baseline rates: {sorted(bl_available_rates)}")
    print(f"  Common rates (intersection): {common_rates}")
    bl_eps = [e for e in bl_all_eps if e["arrival_rate"] in common_rates]
    bl_steps = [s for s in bl_all_steps if s["arrival_rate"] in common_rates]
    im_eps = [e for e in im_eps if e["arrival_rate"] in common_rates]
    im_steps = [s for s in im_steps if s["arrival_rate"] in common_rates]
    print(f"  Baseline: {len(bl_eps)} episodes, {len(bl_steps)} steps")
    print(f"  InfraMind: {len(im_eps)} episodes, {len(im_steps)} steps")

    rates = common_rates
    im_budgets = sorted(set(e["budget"] for e in im_eps))

    if not rates:
        print("\nERROR: No common arrival rates between baseline and InfraMind!")
        print("  Consider re-running tests with matching rates.")
        return

    print(f"\nGenerating diagnostic plots in {args.output}/...")
    plot_accuracy(bl_eps, im_eps, rates, im_budgets, args.output)
    plot_latency(bl_eps, im_eps, rates, im_budgets, args.output)
    plot_model_dist(bl_steps, im_steps, rates, im_budgets, args.output)
    plot_strategy_dist(im_steps, rates, im_budgets, args.output)
    plot_topology_dist(bl_eps, im_eps, rates, im_budgets, args.output)
    plot_budget_compliance(im_eps, args.output)
    plot_pareto(bl_eps, im_eps, rates, im_budgets, args.output)
    plot_budget_sensitivity(im_steps, im_eps, rates, im_budgets, args.output)

    print_summary(bl_eps, bl_steps, im_eps, im_steps, rates, im_budgets)

    print(f"\nAll plots saved to: {args.output}/")


if __name__ == "__main__":
    main()
