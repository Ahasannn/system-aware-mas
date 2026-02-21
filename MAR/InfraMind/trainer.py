"""
InfraMind Trainer — two-level training for hierarchical CMDP router.

Quality-first reward with effort mandate and dense shaping.

Planner training (REINFORCE with normalized advantages):
    correct: utility = 1.0 - min(0.5, max(0, L/B - 1) * 0.3)     → [0.50, 1.0]
    wrong:   utility = -1.0 + 0.3 * min(1, L/B)                   → [-1.0, -0.7]
    advantage = (utility - mean) / std
    loss = -log_prob * advantage + task_loss + vae_loss * 0.001

Executor training (Actor-Critic):
    correct: reward = 1.0 - min(0.5, max(0, L/B - 1) * 0.3)      → [0.50, 1.0]
    wrong:   reward = -1.0 + 0.3 * quality_predictor_score         → [-1.0, -0.7]
             (falls back to effort = L/B if quality predictor unavailable)
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

Correct ALWAYS outranks wrong (min gap = 1.20).
Effort mandate: wrong+tried_hard > wrong+gave_up.
Quality predictor provides dense gradient within wrong answers.
"""

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger

from MAR.InfraMind.inframind_router import InfraMindRouter


class InfraMindTrainer:
    """Two-level trainer for InfraMindRouter."""

    def __init__(
        self,
        router: InfraMindRouter,
        lr_planner: float = 3e-4,
        lr_executor: float = 3e-4,
        executor_entropy_coef: float = 0.10,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.router = router
        self.executor_entropy_coef = executor_entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Planner optimizer (MAS modules)
        self.planner_optimizer = torch.optim.Adam(
            router.planner_parameters(), lr=lr_planner,
        )
        # Executor optimizer (MLP backbone + heads)
        self.executor_optimizer = torch.optim.Adam(
            router.executor_parameters(), lr=lr_executor,
        )

        # LR schedulers (attached via attach_lr_schedulers)
        self.planner_scheduler: Optional[ReduceLROnPlateau] = None
        self.executor_scheduler: Optional[ReduceLROnPlateau] = None

    def attach_lr_schedulers(self, patience: int = 2, factor: float = 0.5) -> None:
        """Attach ReduceLROnPlateau schedulers to both optimizers."""
        self.planner_scheduler = ReduceLROnPlateau(
            self.planner_optimizer, mode="max", patience=patience, factor=factor,
        )
        self.executor_scheduler = ReduceLROnPlateau(
            self.executor_optimizer, mode="max", patience=patience, factor=factor,
        )
        logger.info("LR schedulers attached (patience={}, factor={})", patience, factor)

    def step_schedulers(self, val_solve_rate: float) -> None:
        """Step LR schedulers with validation solve rate."""
        if self.planner_scheduler is not None:
            self.planner_scheduler.step(val_solve_rate)
        if self.executor_scheduler is not None:
            self.executor_scheduler.step(val_solve_rate)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train_batch(
        self,
        planner_transitions: Iterable[dict],
        executor_transitions: Iterable[dict],
        task_label: int = 0,
    ) -> Dict[str, float]:
        planner_batch: List[dict] = list(planner_transitions)
        executor_batch: List[dict] = list(executor_transitions)
        metrics: Dict[str, float] = {}

        if planner_batch:
            metrics.update(self._train_planner(planner_batch, task_label))
            # Compute cost_ratio for logging
            lats = [float(item.get("workflow_latency_seconds", 0.0)) for item in planner_batch]
            buds = [float(item.get("budget_total", 60.0)) for item in planner_batch]
            ratios = [l / max(b, 1.0) for l, b in zip(lats, buds)]
            metrics["cost_ratio"] = sum(ratios) / len(ratios)
            metrics["constraint_gap"] = metrics["cost_ratio"] - 1.0
        if executor_batch:
            metrics.update(self._train_executor(executor_batch))
        return metrics

    # ------------------------------------------------------------------
    # Planner training (MAS-style REINFORCE)
    # ------------------------------------------------------------------

    def _train_planner(self, batch: List[dict], task_label: int = 0) -> Dict[str, float]:
        """
        Quality-first REINFORCE with effort mandate:
            correct: utility = 1.0 - min(0.5, max(0, L/B - 1) * 0.3)  → [0.50, 1.0]
            wrong:   utility = -1.0 + 0.3 * min(1, L/B)               → [-1.0, -0.7]
            advantage = (utility - mean) / std  (normalized)
            loss = -log_prob * advantage + task_loss + vae_loss * 0.001

        Correct ALWAYS outranks wrong. Wrong+tried_hard > wrong+gave_up.
        """
        device = self.router.device

        # Accumulate per-item data across the batch
        log_probs = []
        vae_losses = []
        task_losses = []
        utilities = []

        tasks_y = torch.tensor([task_label], device=device)

        for item in batch:
            # Re-compute forward pass with CURRENT weights (fresh graph)
            fresh = self.router.evaluate_plan(
                query_embedding=item["query_embedding"],
                chosen_topology_idx=item["chosen_topology_idx"],
                chosen_role_indices=item["chosen_role_indices"],
                agent_count=item["agent_count"],
            )
            log_prob = fresh["planner_log_probs"]   # (1, 1) — fresh graph
            vae_loss = fresh["planner_vae_loss"]     # scalar — fresh graph
            task_probs = fresh["task_probs"]          # (1, N_tasks) — fresh graph

            is_solved = float(item.get("is_solved", 0))
            latency = float(item["workflow_latency_seconds"])
            budget = float(item["budget_total"])

            # Quality-first utility with effort mandate
            # Correct → [0.50, 1.0], Wrong → [-1.0, -0.7]
            overshoot = max(0.0, latency / max(budget, 1e-3) - 1.0)
            penalty = min(0.5, overshoot * 0.3)
            if is_solved > 0.5:
                utility = 1.0 - penalty
            else:
                effort = min(1.0, latency / max(budget, 1e-3))
                utility = -1.0 + 0.3 * effort
            utilities.append(utility)
            log_probs.append(log_prob.squeeze())

            # Task classification loss
            task_loss = F.cross_entropy(task_probs, tasks_y)
            task_losses.append(task_loss)

            # VAE regularization
            vae_losses.append(vae_loss)

        # Normalized REINFORCE: advantage = (utility - mean) / std
        baseline = sum(utilities) / max(len(utilities), 1)
        advantages = [u - baseline for u in utilities]
        if len(advantages) > 1:
            adv_std = (sum(a ** 2 for a in advantages) / len(advantages)) ** 0.5
            if adv_std > 1e-8:
                advantages = [a / adv_std for a in advantages]
        answer_losses = [-lp * adv for lp, adv in zip(log_probs, advantages)]
        answer_loss_mean = torch.stack(answer_losses).mean()
        task_loss_mean = torch.stack(task_losses).mean()
        vae_loss_mean = torch.stack(vae_losses).mean()

        loss = answer_loss_mean + task_loss_mean + vae_loss_mean * 0.001

        self.planner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.router.planner_parameters(), self.max_grad_norm,
        )
        self.planner_optimizer.step()

        avg_utility = sum(utilities) / max(len(utilities), 1)
        return {
            "planner_loss": float(loss.detach().cpu().item()),
            "planner_answer_loss": float(answer_loss_mean.detach().cpu().item()),
            "planner_task_loss": float(task_loss_mean.detach().cpu().item()),
            "planner_vae_loss": float(vae_loss_mean.detach().cpu().item()),
            "planner_avg_utility": avg_utility,
        }

    # ------------------------------------------------------------------
    # Executor training (Actor-Critic)
    # ------------------------------------------------------------------

    def _train_executor(self, batch: List[dict]) -> Dict[str, float]:
        device = self.router.device

        states = torch.stack([item["state"]["state"] for item in batch]).to(device)
        model_idx = torch.tensor(
            [int(item["action"]["model_index"].item()) for item in batch], device=device,
        )
        strategy_idx = torch.tensor(
            [int(item["action"]["strategy_index"].item()) for item in batch], device=device,
        )
        is_solved = torch.tensor([item["quality"] for item in batch], device=device)
        # Step-level latency and budget_remaining for per-step credit assignment
        step_latencies = torch.tensor(
            [float(item.get("latency_seconds", 0.0)) for item in batch],
            device=device,
        )
        budget_remaining = torch.tensor(
            [float(item.get("budget_remaining", item.get("budget_total", 60.0)))
             for item in batch],
            device=device,
        )

        # Quality predictor as step-level credit assignment for BOTH correct and wrong
        quality_weights = None
        if self.router.quality_predictor is not None:
            qw_values = []
            qp_failures = 0
            for item in batch:
                query_text = item.get("query_text", "")
                model_name = item.get("model", "")
                role_name = item.get("role", "")
                strategy_name = item.get("strategy", "")
                response_text = item.get("response", "")
                if query_text and model_name and role_name and strategy_name and response_text:
                    try:
                        q = self.router.quality_predictor.predict_quality(
                            query_text, model_name=model_name,
                            role_name=role_name, strategy_name=strategy_name,
                            response=response_text,
                        )
                        qw_values.append(q / 10.0)
                    except Exception:
                        qw_values.append(0.5)  # neutral credit on failure
                        qp_failures += 1
                else:
                    qw_values.append(0.5)  # neutral credit when data missing
                    qp_failures += 1
            if qp_failures > 0:
                logger.warning(
                    "[Trainer] Quality predictor failed for {}/{} executor transitions",
                    qp_failures, len(batch),
                )
            quality_weights = torch.tensor(qw_values, device=device, dtype=torch.float32)

        rewards = self.router.compute_executor_reward(
            is_solved, step_latencies, budget_remaining, quality_weights=quality_weights,
        ).detach()

        # Forward pass on full batch
        action_out = self.router.get_executor_action(states, deterministic=False)
        log_model = torch.log(action_out["model_probs"] + 1e-8)
        log_strategy = torch.log(action_out["strategy_probs"] + 1e-8)
        chosen_log_prob = (
            log_model.gather(1, model_idx.unsqueeze(-1)).squeeze(-1)
            + log_strategy.gather(1, strategy_idx.unsqueeze(-1)).squeeze(-1)
        )
        entropy = (
            -(action_out["model_probs"] * torch.log(action_out["model_probs"] + 1e-8)).sum(dim=1)
            + -(action_out["strategy_probs"] * torch.log(action_out["strategy_probs"] + 1e-8)).sum(dim=1)
        )
        advantage = rewards - action_out["value"].detach()
        # Normalize advantages to zero mean, unit variance
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_loss = -(chosen_log_prob * advantage).mean()
        value_loss = F.mse_loss(action_out["value"], rewards)
        loss = actor_loss + self.value_coef * value_loss - self.executor_entropy_coef * entropy.mean()

        self.executor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.router.executor_parameters(), self.max_grad_norm,
        )
        self.executor_optimizer.step()

        return {
            "executor_loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "executor_entropy": float(entropy.mean().detach().cpu().item()),
        }
