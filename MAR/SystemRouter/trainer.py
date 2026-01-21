from typing import Iterable, List, Dict

import torch
import torch.nn.functional as F

from MAR.SystemRouter.system_aware_router import SystemAwareRouter


class SystemRouterTrainer:
    """
    Two-level REINFORCE-style trainer for the hierarchical router.
    """

    def __init__(
        self,
        router: SystemAwareRouter,
        lr_planner: float = 3e-4,
        lr_executor: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        lambda_lr: float = 1e-3,
    ) -> None:
        self.router = router
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lambda_lr = lambda_lr
        self.planner_optimizer = torch.optim.Adam(
            list(router.planner_backbone.parameters())
            + list(router.topology_head.parameters())
            + list(router.role_head.parameters()),
            lr=lr_planner,
        )
        self.executor_optimizer = torch.optim.Adam(
            list(router.executor_backbone.parameters())
            + list(router.model_head.parameters())
            + list(router.strategy_head.parameters())
            + list(router.value_head.parameters()),
            lr=lr_executor,
        )

    def train_batch(self, planner_transitions: Iterable[dict], executor_transitions: Iterable[dict]) -> Dict[str, float]:
        planner_batch: List[dict] = list(planner_transitions)
        executor_batch: List[dict] = list(executor_transitions)
        metrics: Dict[str, float] = {}

        if planner_batch:
            metrics.update(self._train_planner(planner_batch))
        if executor_batch:
            metrics.update(self._train_executor(executor_batch))
        return metrics

    def _train_planner(self, batch: List[dict]) -> Dict[str, float]:
        device = self.router.device
        states = torch.stack([item["state"] for item in batch]).to(device)
        topology_idx = torch.tensor([int(item["action"]["topology_index"].item()) for item in batch], device=device)
        role_idx = torch.tensor([int(item["action"]["role_index"].item()) for item in batch], device=device)
        rewards = torch.tensor([item["quality"] for item in batch], device=device)

        outputs = self.router.planner_forward(states)
        log_topology = torch.log(outputs["topology_probs"] + 1e-8)
        log_role = torch.log(outputs["role_probs"] + 1e-8)
        chosen_log_prob = (
            log_topology.gather(1, topology_idx.unsqueeze(-1)).squeeze(-1)
            + log_role.gather(1, role_idx.unsqueeze(-1)).squeeze(-1)
        )
        entropy = (
            -(outputs["topology_probs"] * torch.log(outputs["topology_probs"] + 1e-8)).sum(dim=1)
            -(outputs["role_probs"] * torch.log(outputs["role_probs"] + 1e-8)).sum(dim=1)
        )

        advantage = rewards - rewards.mean()
        loss = -(chosen_log_prob * advantage.detach()).mean() - self.entropy_coef * entropy.mean()

        self.planner_optimizer.zero_grad()
        loss.backward()
        self.planner_optimizer.step()

        return {"planner_loss": float(loss.detach().cpu().item())}

    def _train_executor(self, batch: List[dict]) -> Dict[str, float]:
        device = self.router.device
        states = torch.stack([item["state"]["state"] for item in batch]).to(device)
        model_idx = torch.tensor([int(item["action"]["model_index"].item()) for item in batch], device=device)
        strategy_idx = torch.tensor([int(item["action"]["strategy_index"].item()) for item in batch], device=device)
        qualities = torch.tensor([item["quality"] for item in batch], device=device)
        latencies = torch.tensor([item["latency_seconds"] for item in batch], device=device)
        budgets = torch.tensor([item["budget_remaining"] for item in batch], device=device)

        action_out = self.router.get_executor_action(states, deterministic=False)
        rewards = self.router.compute_executor_reward(qualities, latencies, budgets)

        log_model = torch.log(action_out["model_probs"] + 1e-8)
        log_strategy = torch.log(action_out["strategy_probs"] + 1e-8)
        chosen_log_prob = (
            log_model.gather(1, model_idx.unsqueeze(-1)).squeeze(-1)
            + log_strategy.gather(1, strategy_idx.unsqueeze(-1)).squeeze(-1)
        )
        entropy = (
            -(action_out["model_probs"] * torch.log(action_out["model_probs"] + 1e-8)).sum(dim=1)
            -(action_out["strategy_probs"] * torch.log(action_out["strategy_probs"] + 1e-8)).sum(dim=1)
        )
        advantage = rewards - action_out["value"].detach()
        actor_loss = -(chosen_log_prob * advantage).mean()
        value_loss = F.mse_loss(action_out["value"], rewards)
        loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        self.executor_optimizer.zero_grad()
        loss.backward()
        self.executor_optimizer.step()

        constraint_violation = torch.relu(latencies - budgets).mean().detach()
        with torch.no_grad():
            self.router.lagrange_multiplier.add_(self.lambda_lr * constraint_violation)
            self.router.lagrange_multiplier.clamp_(0.0, 10.0)

        return {
            "executor_loss": float(loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "constraint_violation": float(constraint_violation.detach().cpu().item()),
        }
