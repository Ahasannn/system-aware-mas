from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from MAR.SystemRouter.latency_estimator.data import LatencyEstimatorMetadata
from MAR.SystemRouter.latency_estimator.model import LatencyEstimator, LatencyEstimatorConfig


@dataclass(frozen=True)
class LatencyEstimatorBundle:
    model: LatencyEstimator
    metadata: LatencyEstimatorMetadata

    def predict_latency(
        self,
        *,
        prompt_len: float,
        waiting_queue: float,
        running_queue: float,
        kv_cache_usage: float,
        avg_tpot: float,
        avg_ttft: float,
        model_name: str,
        role_name: str,
        strategy_name: str,
    ) -> Tuple[float, float]:
        model_id = self.metadata.model_vocab.encode(model_name)
        role_id = self.metadata.role_vocab.encode(role_name)
        strategy_id = self.metadata.strategy_vocab.encode(strategy_name)
        return self.predict_latency_from_ids(
            prompt_len=prompt_len,
            waiting_queue=waiting_queue,
            running_queue=running_queue,
            kv_cache_usage=kv_cache_usage,
            avg_tpot=avg_tpot,
            avg_ttft=avg_ttft,
            model_id=model_id,
            role_id=role_id,
            strategy_id=strategy_id,
        )

    def predict_latency_from_ids(
        self,
        *,
        prompt_len: float,
        waiting_queue: float,
        running_queue: float,
        kv_cache_usage: float,
        avg_tpot: float,
        avg_ttft: float,
        model_id: int,
        role_id: int,
        strategy_id: int,
    ) -> Tuple[float, float]:
        device = next(self.model.parameters()).device
        x_num = torch.tensor(
            [
                [
                    float(prompt_len),
                    float(waiting_queue),
                    float(running_queue),
                    float(kv_cache_usage),
                    float(avg_tpot),
                    float(avg_ttft),
                ]
            ],
            device=device,
            dtype=torch.float32,
        )
        strategy_tensor = torch.tensor([strategy_id], device=device)
        role_tensor = torch.tensor([role_id], device=device)
        model_tensor = torch.tensor([model_id], device=device)
        self.model.eval()
        with torch.no_grad():
            ttft, tpot = self.model(x_num, strategy_tensor, role_tensor, model_tensor)
        return float(ttft.item()), float(tpot.item())


def save_latency_estimator(
    path: Union[str, Path],
    model: LatencyEstimator,
    metadata: LatencyEstimatorMetadata,
    config: LatencyEstimatorConfig,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata.to_dict(),
        "config": config.to_dict(),
    }
    torch.save(payload, str(path))


def load_latency_estimator(
    path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[LatencyEstimator, LatencyEstimatorMetadata, LatencyEstimatorConfig]:
    payload = torch.load(str(path), map_location=device or "cpu")
    metadata = LatencyEstimatorMetadata.from_dict(payload["metadata"])
    config = LatencyEstimatorConfig.from_dict(payload["config"])
    model = LatencyEstimator(
        num_numerical_features=config.num_numerical_features,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dims=list(config.hidden_dims),
        dropout=config.dropout,
    )
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(device)
    return model, metadata, config
