"""
InfraMind Trainer — two-level training for hierarchical CMDP router.

Planner training (MAS-style REINFORCE):
    utility = is_solved - (latency / budget) * latency_rate
    loss = -log_prob * utility + task_loss + vae_loss * 0.001

Executor training (CMDP Actor-Critic):
    reward = quality - λ · (step_latency / budget_remaining)
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

Lambda update (episode-level Lagrangian dual ascent):
    λ ← λ + lr · clamp(mean(workflow_lat / budget_total) - 1, -1, 2)
"""

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from MAR.InfraMind.inframind_router import InfraMindRouter


class InfraMindTrainer:
    """Two-level trainer for InfraMindRouter."""

    def __init__(
        self,
        router: InfraMindRouter,
        lr_planner: float = 3e-4,
        lr_executor: float = 3e-4,
        latency_rate: float = 1.0,
        executor_entropy_coef: float = 0.10,
        value_coef: float = 0.5,
        lambda_lr: float = 5e-3,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.router = router
        self.latency_rate = latency_rate
        self.executor_entropy_coef = executor_entropy_coef
        self.value_coef = value_coef
        self.lambda_lr = lambda_lr
        self.max_grad_norm = max_grad_norm

        # Planner optimizer (MAS modules + BCFM)
        self.planner_optimizer = torch.optim.Adam(
            router.planner_parameters(), lr=lr_planner,
        )
        # Executor optimizer (MLP backbone + heads)
        self.executor_optimizer = torch.optim.Adam(
            router.executor_parameters(), lr=lr_executor,
        )

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
            metrics.update(self._update_lambda(planner_batch))
        if executor_batch:
            metrics.update(self._train_executor(executor_batch))
        return metrics

    # ------------------------------------------------------------------
    # Lambda update (episode-level Lagrange dual ascent)
    # ------------------------------------------------------------------

    def _update_lambda(self, planner_batch: List[dict]) -> Dict[str, float]:
        """Episode-level constraint update: lambda += lr * (mean(workflow_lat / budget_total) - 1).

        Uses planner transitions which carry episode-level workflow_latency_seconds
        and budget_total, not per-step values.  The constraint gap is clamped to
        [-1, 2] to prevent single-batch explosion.
        """
        with torch.no_grad():
            lats = [float(item.get("workflow_latency_seconds", 0.0)) for item in planner_batch]
            buds = [float(item.get("budget_total", 60.0)) for item in planner_batch]
            ratios = [l / max(b, 1.0) for l, b in zip(lats, buds)]
            cost_ratio = sum(ratios) / len(ratios)
            constraint_gap = cost_ratio - 1.0
            # Clamp gap to prevent single-batch explosion
            constraint_gap = max(-1.0, min(constraint_gap, 2.0))
            self.router.lagrange_multiplier.add_(self.lambda_lr * constraint_gap)
            self.router.lagrange_multiplier.clamp_(-5.0, 3.0)  # softplus(3) ≈ 3.05
        return {
            "cost_ratio": cost_ratio,
            "constraint_gap": constraint_gap,
            "lambda": float(F.softplus(self.router.lagrange_multiplier).item()),
        }

    # ------------------------------------------------------------------
    # Planner training (MAS-style REINFORCE)
    # ------------------------------------------------------------------

    def _train_planner(self, batch: List[dict], task_label: int = 0) -> Dict[str, float]:
        """
        MAS-style REINFORCE with latency cost:
            utility = is_solved - (latency / budget) * latency_rate
            answer_loss = -log_prob * utility
            loss = answer_loss + task_loss + vae_loss * 0.001

        Re-computes the planner forward pass with current weights via
        ``router.evaluate_plan()`` to avoid stale computation graphs.
        """
        device = self.router.device

        # Accumulate losses across the batch
        answer_losses = []
        vae_losses = []
        task_losses = []
        utilities = []

        tasks_y = torch.tensor([task_label], device=device)

        for item in batch:
            # Re-compute forward pass with CURRENT weights (fresh graph)
            fresh = self.router.evaluate_plan(
                query_embedding=item["query_embedding"],
                budget_total=item["budget_total"],
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

            # Utility: is_solved ∈ {0,1} minus normalized latency cost
            cost = (latency / max(budget, 1e-3)) * self.latency_rate
            utility = is_solved - cost
            utilities.append(utility)

            # REINFORCE loss
            answer_loss = -log_prob.squeeze() * utility
            answer_losses.append(answer_loss)

            # Task classification loss
            task_loss = F.cross_entropy(task_probs, tasks_y)
            task_losses.append(task_loss)

            # VAE regularization
            vae_losses.append(vae_loss)

        # Average losses
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
    # Executor training (CMDP Actor-Critic)
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
        qualities = torch.tensor([item["quality"] for item in batch], device=device)
        latencies = torch.tensor(
            [float(item.get("latency_seconds", 0.0)) for item in batch],
            device=device,
        )
        budgets = torch.tensor([item["budget_remaining"] for item in batch], device=device)

        # Predict quality from quality predictor if available (now requires response)
        predicted_quality = None
        if self.router.quality_predictor is not None:
            pq_values = []
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
                        pq_values.append(q)
                    except Exception:
                        pq_values.append(5.0)  # neutral fallback
                else:
                    pq_values.append(5.0)
            predicted_quality = torch.tensor(pq_values, device=device, dtype=torch.float32)

        rewards = self.router.compute_executor_reward(
            qualities, latencies, budgets, model_index=model_idx,
            predicted_quality=predicted_quality,
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
        # Normalize advantages to zero mean, unit variance — standard practice
        # that prevents absolute reward magnitude from dominating the gradient.
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
            "lambda": float(F.softplus(self.router.lagrange_multiplier).item()),
            "executor_entropy": float(entropy.mean().detach().cpu().item()),
        }
