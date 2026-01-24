from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from MAR.SystemRouter.latency_estimator.vocab import CategoricalVocab


DEFAULT_NUM_FEATURES = (
    "prompt_len",
    "waiting_queue",
    "running_queue",
    "kv_cache_usage",
    "avg_tpot",
    "avg_ttft",
)


@dataclass(frozen=True)
class LatencyEstimatorRecord:
    prompt_len: float
    waiting_queue: float
    running_queue: float
    kv_cache_usage: float
    avg_tpot: float
    avg_ttft: float
    model_name: str
    role_name: str
    strategy_name: str
    ttft: float
    tpot: float


@dataclass(frozen=True)
class LatencyEstimatorMetadata:
    model_vocab: CategoricalVocab
    role_vocab: CategoricalVocab
    strategy_vocab: CategoricalVocab
    num_numerical_features: int
    feature_names: Tuple[str, ...] = DEFAULT_NUM_FEATURES

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_vocab": self.model_vocab.to_dict(),
            "role_vocab": self.role_vocab.to_dict(),
            "strategy_vocab": self.strategy_vocab.to_dict(),
            "num_numerical_features": self.num_numerical_features,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LatencyEstimatorMetadata":
        feature_names = tuple(data.get("feature_names", DEFAULT_NUM_FEATURES))
        return cls(
            model_vocab=CategoricalVocab.from_dict(data["model_vocab"]),
            role_vocab=CategoricalVocab.from_dict(data["role_vocab"]),
            strategy_vocab=CategoricalVocab.from_dict(data["strategy_vocab"]),
            num_numerical_features=int(data.get("num_numerical_features", len(feature_names))),
            feature_names=feature_names,
        )


class LatencyEstimatorDataset(Dataset):
    def __init__(
        self,
        records: Sequence[LatencyEstimatorRecord],
        metadata: LatencyEstimatorMetadata,
    ) -> None:
        self.records = list(records)
        self.metadata = metadata
        self.num_features = metadata.num_numerical_features

        self.x_num = torch.tensor(
            [
                [
                    record.prompt_len,
                    record.waiting_queue,
                    record.running_queue,
                    record.kv_cache_usage,
                    record.avg_tpot,
                    record.avg_ttft,
                ]
                for record in self.records
            ],
            dtype=torch.float32,
        )
        self.strategy_ids = torch.tensor(
            [metadata.strategy_vocab.encode(record.strategy_name) for record in self.records],
            dtype=torch.long,
        )
        self.role_ids = torch.tensor(
            [metadata.role_vocab.encode(record.role_name) for record in self.records],
            dtype=torch.long,
        )
        self.model_ids = torch.tensor(
            [metadata.model_vocab.encode(record.model_name) for record in self.records],
            dtype=torch.long,
        )
        self.targets_ttft = torch.tensor([record.ttft for record in self.records], dtype=torch.float32)
        self.targets_tpot = torch.tensor([record.tpot for record in self.records], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x_num[idx],
            self.strategy_ids[idx],
            self.role_ids[idx],
            self.model_ids[idx],
            self.targets_ttft[idx],
            self.targets_tpot[idx],
        )


def load_latency_records_from_csv(
    csv_path: Union[str, Path],
    *,
    record_type: Optional[Union[str, Sequence[str]]] = "role_step",
    prompt_tokens_field: str = "prompt_tokens",
    prompt_fields: Sequence[str] = ("prompt_base", "prompt", "query"),
    waiting_field: str = "llm_waiting",
    running_field: str = "llm_running",
    kv_field: str = "llm_kv_cache_usage",
    avg_tpot_field: str = "llm_itl_avg",
    avg_ttft_field: str = "llm_ttft_avg",
    ttft_field: str = "observed_ttft",
    tpot_field: str = "observed_tpot",
    model_field: str = "model_name",
    role_field: str = "role_name",
    strategy_field: str = "strategy_name",
    min_ttft: float = 1e-6,
    min_tpot: float = 1e-6,
) -> List[LatencyEstimatorRecord]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    record_types: Optional[set] = None
    if record_type:
        if isinstance(record_type, str):
            record_types = {record_type}
        else:
            record_types = {str(value) for value in record_type}

    records: List[LatencyEstimatorRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if record_types is not None:
                row_type = _safe_str(row.get("record_type"))
                if row_type not in record_types:
                    continue

            model_name = _safe_str(row.get(model_field)) or _safe_str(row.get("llm_name"))
            role_name = _safe_str(row.get(role_field))
            strategy_name = _safe_str(row.get(strategy_field))
            if not model_name or not role_name or not strategy_name:
                continue

            prompt_len = _safe_float(row.get(prompt_tokens_field))
            if prompt_len is None:
                prompt_text = ""
                for field in prompt_fields:
                    prompt_text = _safe_str(row.get(field))
                    if prompt_text:
                        break
                prompt_len = _approx_tokens(prompt_text)
            if prompt_len is None:
                continue

            waiting_queue = _safe_float(row.get(waiting_field)) or 0.0
            running_queue = _safe_float(row.get(running_field)) or 0.0
            kv_cache_usage = _safe_float(row.get(kv_field)) or 0.0
            avg_tpot = _safe_float(row.get(avg_tpot_field)) or 0.0
            avg_ttft = _safe_float(row.get(avg_ttft_field)) or 0.0

            ttft = _safe_float(row.get(ttft_field)) or _safe_float(row.get("ttft"))
            tpot = _safe_float(row.get(tpot_field)) or _safe_float(row.get("tpot"))
            if ttft is None or tpot is None:
                continue
            if ttft < min_ttft or tpot < min_tpot:
                continue

            records.append(
                LatencyEstimatorRecord(
                    prompt_len=float(prompt_len),
                    waiting_queue=float(waiting_queue),
                    running_queue=float(running_queue),
                    kv_cache_usage=float(kv_cache_usage),
                    avg_tpot=float(avg_tpot),
                    avg_ttft=float(avg_ttft),
                    model_name=model_name,
                    role_name=role_name,
                    strategy_name=strategy_name,
                    ttft=float(ttft),
                    tpot=float(tpot),
                )
            )
    return records


def build_latency_estimator_metadata(
    records: Sequence[LatencyEstimatorRecord],
    *,
    feature_names: Sequence[str] = DEFAULT_NUM_FEATURES,
    unk_token: str = "<unk>",
) -> LatencyEstimatorMetadata:
    model_vocab = _build_vocab([record.model_name for record in records], unk_token=unk_token)
    role_vocab = _build_vocab([record.role_name for record in records], unk_token=unk_token)
    strategy_vocab = _build_vocab([record.strategy_name for record in records], unk_token=unk_token)
    return LatencyEstimatorMetadata(
        model_vocab=model_vocab,
        role_vocab=role_vocab,
        strategy_vocab=strategy_vocab,
        num_numerical_features=len(feature_names),
        feature_names=tuple(feature_names),
    )


def prepare_latency_estimator_dataset(
    csv_path: Union[str, Path],
    *,
    record_type: Optional[Union[str, Sequence[str]]] = "role_step",
    min_ttft: float = 1e-6,
    min_tpot: float = 1e-6,
) -> Tuple[LatencyEstimatorDataset, LatencyEstimatorMetadata]:
    records = load_latency_records_from_csv(
        csv_path,
        record_type=record_type,
        min_ttft=min_ttft,
        min_tpot=min_tpot,
    )
    metadata = build_latency_estimator_metadata(records)
    dataset = LatencyEstimatorDataset(records, metadata)
    return dataset, metadata


def _build_vocab(values: Iterable[str], *, unk_token: str = "<unk>") -> CategoricalVocab:
    tokens = sorted({value for value in values if value})
    if unk_token in tokens:
        tokens.remove(unk_token)
    tokens.insert(0, unk_token)
    return CategoricalVocab(tokens, unk_token=unk_token)


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _approx_tokens(text: str) -> Optional[float]:
    cleaned = text.strip()
    if not cleaned:
        return None
    return float(len(cleaned.split()))
