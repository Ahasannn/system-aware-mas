from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LatencyEstimatorConfig:
    num_numerical_features: int = 8
    embedding_dim: int = 16
    hidden_dims: List[int] = None
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            object.__setattr__(self, "hidden_dims", [128, 64])

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_numerical_features": self.num_numerical_features,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LatencyEstimatorConfig":
        return cls(
            num_numerical_features=int(data.get("num_numerical_features", 8)),
            embedding_dim=int(data.get("embedding_dim", 16)),
            hidden_dims=list(data.get("hidden_dims", [128, 64])),
            dropout=float(data.get("dropout", 0.1)),
        )


class LatencyEstimator(nn.Module):
    """
    Multi-head MLP regressor for ttft/tpot prediction.
    """

    def __init__(
        self,
        *,
        num_numerical_features: int = 8,
        num_strategies: int = 3,
        num_roles: int = 10,
        num_models: int = 5,
        embedding_dim: int = 16,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two layer sizes.")

        self.num_features = num_numerical_features
        self.feature_norm = nn.BatchNorm1d(num_numerical_features)
        self.strategy_embedding = nn.Embedding(num_strategies, embedding_dim)
        self.role_embedding = nn.Embedding(num_roles, embedding_dim)
        self.model_embedding = nn.Embedding(num_models, embedding_dim)

        fused_dim = num_numerical_features + 3 * embedding_dim
        self.backbone = nn.Sequential(
            nn.Linear(fused_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_ttft = nn.Linear(hidden_dims[1], 1)
        self.head_tpot = nn.Linear(hidden_dims[1], 1)

    def forward(
        self,
        x_num: torch.Tensor,
        strategy_ids: torch.Tensor,
        role_ids: torch.Tensor,
        model_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_num.dim() == 1:
            x_num = x_num.unsqueeze(0)
        x_num = self.feature_norm(x_num)
        strategy_ids = strategy_ids.long().view(-1)
        role_ids = role_ids.long().view(-1)
        model_ids = model_ids.long().view(-1)

        fused = torch.cat(
            [
                x_num,
                self.strategy_embedding(strategy_ids),
                self.role_embedding(role_ids),
                self.model_embedding(model_ids),
            ],
            dim=-1,
        )
        hidden = self.backbone(fused)
        ttft = F.softplus(self.head_ttft(hidden)).squeeze(-1)
        tpot = F.softplus(self.head_tpot(hidden)).squeeze(-1)
        return ttft, tpot
