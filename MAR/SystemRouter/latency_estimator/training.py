from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from MAR.SystemRouter.latency_estimator.data import LatencyEstimatorDataset
from MAR.SystemRouter.latency_estimator.model import LatencyEstimator


def train_latency_estimator(
    model: LatencyEstimator,
    dataset: LatencyEstimatorDataset,
    *,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    epoch_losses: List[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x_num, strategy_id, role_id, model_id, ttft_target, tpot_target in loader:
            x_num = x_num.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            ttft_target = ttft_target.to(device)
            tpot_target = tpot_target.to(device)

            optimizer.zero_grad()
            ttft_pred, tpot_pred = model(x_num, strategy_id, role_id, model_id)
            loss = loss_fn(ttft_pred, ttft_target) + loss_fn(tpot_pred, tpot_target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * x_num.size(0)
            total_items += x_num.size(0)
        epoch_losses.append(total_loss / max(total_items, 1))
    return epoch_losses
