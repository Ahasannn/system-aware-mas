"""Quality Estimator — predicts LLM-as-a-Judge score from (query, response, model, role, strategy).

Mirrors the architecture in ``length_estimator.py`` but outputs a bounded
quality score in [0, 10] via ``sigmoid * 10``.  Both the prompt and the LLM
response are embedded via SentenceTransformer and compressed before fusion.

Training target: ``judge_score`` column from the judge-scored CSV.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from MAR.InfraMind.inframind_router import SemanticEncoder
from MAR.InfraMind.length_estimator import CategoricalVocab, encode_prompts


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityEstimatorRecord:
    prompt: str
    model_name: str
    role_name: str
    strategy_name: str
    judge_score: float
    response: str = ""  # LLM response text (used as additional input feature)
    item_id: str = ""  # grouping key for train/val/test split (e.g. query text)


@dataclass(frozen=True)
class QualityEstimatorMetadata:
    model_vocab: CategoricalVocab
    role_vocab: CategoricalVocab
    strategy_vocab: CategoricalVocab
    semantic_dim: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_vocab": self.model_vocab.to_dict(),
            "role_vocab": self.role_vocab.to_dict(),
            "strategy_vocab": self.strategy_vocab.to_dict(),
            "semantic_dim": self.semantic_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "QualityEstimatorMetadata":
        return cls(
            model_vocab=CategoricalVocab.from_dict(data["model_vocab"]),
            role_vocab=CategoricalVocab.from_dict(data["role_vocab"]),
            strategy_vocab=CategoricalVocab.from_dict(data["strategy_vocab"]),
            semantic_dim=int(data["semantic_dim"]),
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityEstimatorConfig:
    semantic_dim: int = 384
    semantic_compress_dim: int = 64
    response_compress_dim: int = 64
    embedding_dim: int = 16
    mlp_hidden: Tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, object]:
        return {
            "semantic_dim": self.semantic_dim,
            "semantic_compress_dim": self.semantic_compress_dim,
            "response_compress_dim": self.response_compress_dim,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden": list(self.mlp_hidden),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "QualityEstimatorConfig":
        return cls(
            semantic_dim=int(data.get("semantic_dim", 384)),
            semantic_compress_dim=int(data.get("semantic_compress_dim", 64)),
            response_compress_dim=int(data.get("response_compress_dim", 64)),
            embedding_dim=int(data.get("embedding_dim", 16)),
            mlp_hidden=tuple(int(v) for v in data.get("mlp_hidden", (128, 64, 32))),
            dropout=float(data.get("dropout", 0.1)),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QualityEstimator(nn.Module):
    """MLP that predicts judge quality score in [0, 10].

    Input: prompt_emb(384->64) + response_emb(384->64) + strategy_emb(16) + role_emb(16) + model_emb(16) = 176-dim
    Output: sigmoid(linear(32,1)) * 10.0
    """

    def __init__(
        self,
        config: QualityEstimatorConfig,
        *,
        num_models: int,
        num_roles: int,
        num_strategies: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.semantic_compress = nn.Sequential(
            nn.Linear(config.semantic_dim, config.semantic_compress_dim),
            nn.ReLU(),
            nn.LayerNorm(config.semantic_compress_dim),
        )
        self.response_compress = nn.Sequential(
            nn.Linear(config.semantic_dim, config.response_compress_dim),
            nn.ReLU(),
            nn.LayerNorm(config.response_compress_dim),
        )
        self.model_embedding = nn.Embedding(num_models, config.embedding_dim)
        self.role_embedding = nn.Embedding(num_roles, config.embedding_dim)
        self.strategy_embedding = nn.Embedding(num_strategies, config.embedding_dim)

        fused_dim = config.semantic_compress_dim + config.response_compress_dim + 3 * config.embedding_dim
        hidden_1, hidden_2, hidden_3 = config.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_2, hidden_3),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.output_head = nn.Linear(hidden_3, 1)

    def forward(
        self,
        semantic_embeddings: torch.Tensor,
        response_embeddings: torch.Tensor,
        strategy_ids: torch.Tensor,
        role_ids: torch.Tensor,
        model_ids: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_embeddings.dim() == 1:
            semantic_embeddings = semantic_embeddings.unsqueeze(0)
        if response_embeddings.dim() == 1:
            response_embeddings = response_embeddings.unsqueeze(0)
        strategy_ids = strategy_ids.long().view(-1)
        role_ids = role_ids.long().view(-1)
        model_ids = model_ids.long().view(-1)
        semantic_features = self.semantic_compress(semantic_embeddings)
        response_features = self.response_compress(response_embeddings)
        fused = torch.cat(
            [
                semantic_features,
                response_features,
                self.strategy_embedding(strategy_ids),
                self.role_embedding(role_ids),
                self.model_embedding(model_ids),
            ],
            dim=-1,
        )
        hidden = self.mlp(fused)
        # Bounded output in [0, 10]
        output = torch.sigmoid(self.output_head(hidden)).squeeze(-1) * 10.0
        return output


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QualityEstimatorDataset(Dataset):
    def __init__(
        self,
        records: Sequence[QualityEstimatorRecord],
        metadata: QualityEstimatorMetadata,
        semantic_embeddings: torch.Tensor,
        response_embeddings: torch.Tensor,
    ) -> None:
        if len(records) != semantic_embeddings.shape[0]:
            raise ValueError("Records and embeddings must align in length.")
        if len(records) != response_embeddings.shape[0]:
            raise ValueError("Records and response embeddings must align in length.")
        self.records = list(records)
        self.metadata = metadata
        self.semantic_embeddings = semantic_embeddings.float()
        self.response_embeddings = response_embeddings.float()
        self.model_ids = torch.tensor(
            [metadata.model_vocab.encode(record.model_name) for record in self.records],
            dtype=torch.long,
        )
        self.role_ids = torch.tensor(
            [metadata.role_vocab.encode(record.role_name) for record in self.records],
            dtype=torch.long,
        )
        self.strategy_ids = torch.tensor(
            [metadata.strategy_vocab.encode(record.strategy_name) for record in self.records],
            dtype=torch.long,
        )
        self.targets = torch.tensor(
            [float(record.judge_score) for record in self.records],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.semantic_embeddings[idx],
            self.response_embeddings[idx],
            self.strategy_ids[idx],
            self.role_ids[idx],
            self.model_ids[idx],
            self.targets[idx],
        )


# ---------------------------------------------------------------------------
# Bundle (inference wrapper)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityEstimatorBundle:
    model: QualityEstimator
    metadata: QualityEstimatorMetadata
    encoder: SemanticEncoder

    def predict_quality(
        self,
        prompt: str,
        *,
        model_name: str,
        role_name: str,
        strategy_name: str,
        response: str = "",
    ) -> float:
        device = next(self.model.parameters()).device
        semantic = self.encoder([prompt])[0].to(device)
        resp_emb = self.encoder([response or ""])[0].to(device)
        model_id = torch.tensor([self.metadata.model_vocab.encode(model_name)], device=device)
        role_id = torch.tensor([self.metadata.role_vocab.encode(role_name)], device=device)
        strategy_id = torch.tensor([self.metadata.strategy_vocab.encode(strategy_name)], device=device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(semantic.unsqueeze(0), resp_emb.unsqueeze(0), strategy_id, role_id, model_id)
        return float(prediction.item())


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_quality_records_from_csv(
    csv_path: Union[str, Path],
    *,
    record_type: Optional[Union[str, Sequence[str]]] = "role_step",
    prompt_field: str = "prompt_base",
    model_field: str = "model_name",
    role_field: str = "role_name",
    strategy_field: str = "strategy_name",
    score_field: str = "judge_score",
    response_field: str = "response",
) -> List[QualityEstimatorRecord]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    record_types: Optional[set] = None
    if record_type:
        if isinstance(record_type, str):
            record_types = {record_type}
        else:
            record_types = {str(value) for value in record_type}

    records: List[QualityEstimatorRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if record_types is not None:
                row_type = _safe_str(row.get("record_type"))
                if row_type not in record_types:
                    continue
            prompt = _safe_str(row.get(prompt_field)) or _safe_str(row.get("prompt")) or _safe_str(row.get("query"))
            if not prompt:
                continue
            model_name = _safe_str(row.get(model_field)) or _safe_str(row.get("llm_name"))
            role_name = _safe_str(row.get(role_field))
            strategy_name = _safe_str(row.get(strategy_field))
            if not model_name or not role_name or not strategy_name:
                continue
            score_raw = _safe_float(row.get(score_field))
            if score_raw is None:
                continue
            # Clamp to [0, 10]
            score = max(0.0, min(10.0, score_raw))
            response = _safe_str(row.get(response_field)) or _safe_str(row.get("completion"))
            # item_id for grouping: prefer item_id column, fall back to query
            item_id = _safe_str(row.get("item_id")) or _safe_str(row.get("query")) or prompt
            records.append(
                QualityEstimatorRecord(
                    prompt=prompt,
                    model_name=model_name,
                    role_name=role_name,
                    strategy_name=strategy_name,
                    judge_score=score,
                    response=response,
                    item_id=item_id,
                )
            )
    return records


def build_quality_estimator_metadata(
    records: Sequence[QualityEstimatorRecord],
    *,
    semantic_dim: int,
    unk_token: str = "<unk>",
) -> QualityEstimatorMetadata:
    model_vocab = _build_vocab([record.model_name for record in records], unk_token=unk_token)
    role_vocab = _build_vocab([record.role_name for record in records], unk_token=unk_token)
    strategy_vocab = _build_vocab([record.strategy_name for record in records], unk_token=unk_token)
    return QualityEstimatorMetadata(
        model_vocab=model_vocab,
        role_vocab=role_vocab,
        strategy_vocab=strategy_vocab,
        semantic_dim=semantic_dim,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_quality_estimator(
    model: QualityEstimator,
    dataset: QualityEstimatorDataset,
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
        for semantic, response_emb, strategy_id, role_id, model_id, targets in loader:
            semantic = semantic.to(device)
            response_emb = response_emb.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(semantic, response_emb, strategy_id, role_id, model_id)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * semantic.size(0)
            total_items += semantic.size(0)
        epoch_losses.append(total_loss / max(total_items, 1))
    return epoch_losses


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_quality_estimator(
    path: Union[str, Path],
    model: QualityEstimator,
    metadata: QualityEstimatorMetadata,
    config: QualityEstimatorConfig,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata.to_dict(),
        "config": config.to_dict(),
    }
    torch.save(payload, str(path))


def load_quality_estimator(
    path: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[QualityEstimator, QualityEstimatorMetadata, QualityEstimatorConfig]:
    payload = torch.load(str(path), map_location=device or "cpu")
    metadata = QualityEstimatorMetadata.from_dict(payload["metadata"])
    config = QualityEstimatorConfig.from_dict(payload["config"])
    model = QualityEstimator(
        config,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
    )
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(device)
    return model, metadata, config


# ---------------------------------------------------------------------------
# Convenience: prepare dataset in one call
# ---------------------------------------------------------------------------

def prepare_quality_estimator_dataset(
    csv_path: Union[str, Path],
    *,
    encoder: SemanticEncoder,
    record_type: Optional[Union[str, Sequence[str]]] = "role_step",
    batch_size: int = 64,
) -> Tuple[QualityEstimatorDataset, QualityEstimatorMetadata]:
    records = load_quality_records_from_csv(csv_path, record_type=record_type)
    metadata = build_quality_estimator_metadata(records, semantic_dim=encoder.embedding_dim)
    prompt_embeddings = encode_prompts(encoder, [record.prompt for record in records], batch_size=batch_size)
    response_embeddings = encode_prompts(encoder, [record.response or "" for record in records], batch_size=batch_size)
    dataset = QualityEstimatorDataset(records, metadata, prompt_embeddings, response_embeddings)
    return dataset, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_vocab(values: Iterable[str], *, unk_token: str = "<unk>") -> CategoricalVocab:
    tokens = sorted({value for value in values if value})
    if unk_token in tokens:
        tokens.remove(unk_token)
    tokens.insert(0, unk_token)
    return CategoricalVocab(tokens, unk_token=unk_token)


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None
