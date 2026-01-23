from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Tools.reader.readers import JSONLReader

from Datasets.mbpp_dataset import MbppDataset
from Datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from Datasets.math_dataset import MATH_get_predict, MATH_is_correct, load_math_dataset
from Datasets.mmlu_dataset import MMLUDataset


@dataclass(frozen=True)
class SystemRouterSample:
    query: str
    tests: Optional[List[str]]
    answer: Optional[str]
    item_id: object
    metadata: Dict[str, object] = field(default_factory=dict)


def _resolve_item_id(row: object, fallback: int) -> object:
    keys = ("task_id", "item_id", "id", "ID", "index", "idx")
    getter = row.get if hasattr(row, "get") else None
    for key in keys:
        value = None
        if getter is not None:
            value = getter(key)
        elif isinstance(row, dict):
            value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            try:
                if value != value:
                    continue
            except Exception:
                pass
        if value == "":
            continue
        return value
    return fallback


def _extract_code(response: str) -> str:
    import re

    pattern = r"```python(.*?)```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _postprocess_mmlu_answer(answer: Union[str, List[str]]) -> str:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if not isinstance(answer, str):
        raise ValueError("Expected answer as string.")
    if answer:
        ans_pos = answer.find("answer is")
        if ans_pos != -1:
            answer = answer[ans_pos + len("answer is") :].strip(":").strip().strip("Option").strip()
        answer = answer[0]
    return answer


class SystemRouterDataset(ABC):
    dataset_name: str
    role_domain: str
    prompt_file: str

    def __init__(self, split: str, seed: int = 42) -> None:
        self.split = split
        self.seed = seed
        self._samples = self._load_samples()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SystemRouterSample:
        return self._samples[index]

    def sample(self, *, limit: int = 0, shuffle: bool = True, seed: Optional[int] = None) -> List[SystemRouterSample]:
        indices = list(range(len(self._samples)))
        if shuffle:
            rng = random.Random(self.seed if seed is None else seed)
            rng.shuffle(indices)
        if limit and limit > 0:
            indices = indices[: int(limit)]
        return [self._samples[i] for i in indices]

    @abstractmethod
    def _load_samples(self) -> List[SystemRouterSample]:
        raise NotImplementedError

    @abstractmethod
    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[SystemRouterSample],
    ) -> Tuple[float, Dict[str, object]]:
        raise NotImplementedError


class MbppAdapter(SystemRouterDataset):
    dataset_name = "mbpp"
    role_domain = "Code"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "mbpp.json")

    def _load_samples(self) -> List[SystemRouterSample]:
        ds = MbppDataset(self.split)
        samples: List[SystemRouterSample] = []
        for idx, row in ds.df.iterrows():
            query = str(row.get("task") or row["text"])
            tests = list(row["test_list"])
            item_id = _resolve_item_id(row, idx)
            samples.append(SystemRouterSample(query=query, tests=tests, answer=None, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[SystemRouterSample],
    ) -> Tuple[float, Dict[str, object]]:
        if not tests:
            return 0.0, {"is_solved": False, "feedback": "Missing tests."}
        code = _extract_code(response)
        is_solved, feedback, _ = PyExecutor().execute(code, list(tests), timeout=30, verbose=False)
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "feedback": feedback}


class HumanEvalAdapter(SystemRouterDataset):
    dataset_name = "humaneval"
    role_domain = "Code"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "humaneval.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        split_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path or os.path.join("Datasets", "humaneval", "humaneval-py.jsonl")
        self.split_ratio = split_ratio
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[SystemRouterSample]:
        reader = JSONLReader()
        records = reader.parse_file(self.dataset_path)
        indices = list(range(len(records)))
        rng = random.Random(self.seed)
        rng.shuffle(indices)
        split_index = int(len(indices) * self.split_ratio)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        if self.split == "train":
            active = train_indices
        elif self.split == "test":
            active = test_indices
        else:
            active = indices

        samples: List[SystemRouterSample] = []
        for idx in active:
            row = records[idx]
            query = str(row.get("prompt", ""))
            test = row.get("test", "")
            item_id = _resolve_item_id(row, idx)
            tests = [str(test)] if test else []
            samples.append(SystemRouterSample(query=query, tests=tests, answer=None, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[SystemRouterSample],
    ) -> Tuple[float, Dict[str, object]]:
        if not tests:
            return 0.0, {"is_solved": False, "feedback": "Missing tests."}
        code = _extract_code(response)
        is_solved, feedback, _ = PyExecutor().execute(code, list(tests), timeout=30, verbose=False)
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "feedback": feedback}


class Gsm8kAdapter(SystemRouterDataset):
    dataset_name = "gsm8k"
    role_domain = "Math"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "gsm8k.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_path: Optional[str] = None,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        split_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path or os.path.join("Datasets", "gsm8k", "gsm8k.jsonl")
        self.train_path = train_path
        self.test_path = test_path
        self.split_ratio = split_ratio
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[SystemRouterSample]:
        reader = JSONLReader()
        if self.split in ("train", "test") and (self.train_path or self.test_path):
            path = self.train_path if self.split == "train" else self.test_path
            if not path:
                path = self.dataset_path
            records = reader.parse_file(path)
            processed = gsm_data_process(records)
        else:
            records = reader.parse_file(self.dataset_path)
            processed = gsm_data_process(records)
            indices = list(range(len(processed)))
            rng = random.Random(self.seed)
            rng.shuffle(indices)
            split_index = int(len(indices) * self.split_ratio)
            train_indices = indices[:split_index]
            test_indices = indices[split_index:]
            if self.split == "train":
                processed = [processed[i] for i in train_indices]
            elif self.split == "test":
                processed = [processed[i] for i in test_indices]

        samples: List[SystemRouterSample] = []
        for idx, row in enumerate(processed):
            query = str(row.get("task", ""))
            answer = str(row.get("answer", "")).strip()
            item_id = _resolve_item_id(row, idx)
            samples.append(SystemRouterSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[SystemRouterSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = gsm_get_predict(response)
        pred_value = _safe_float(str(pred))
        gold_value = _safe_float(str(answer))
        if pred_value is not None and gold_value is not None:
            is_solved = pred_value == gold_value
        else:
            is_solved = str(pred).strip() == str(answer).strip()
        return float(1.0 if is_solved else 0.0), {"is_solved": bool(is_solved), "pred": pred, "gold": answer}


class MathAdapter(SystemRouterDataset):
    dataset_name = "math"
    role_domain = "Math"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "math.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_root: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.dataset_root = dataset_root or os.path.join("Datasets", "MATH")
        super().__init__(split, seed=seed)

    def _load_samples(self) -> List[SystemRouterSample]:
        records = load_math_dataset(self.dataset_root, split=self.split)
        samples: List[SystemRouterSample] = []
        for idx, row in enumerate(records):
            query = str(row.get("problem", ""))
            answer = str(row.get("solution", ""))
            item_id = _resolve_item_id(row, idx)
            samples.append(SystemRouterSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[SystemRouterSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = MATH_get_predict(response)
        is_solved = bool(MATH_is_correct(pred, answer))
        gold = MATH_get_predict(answer) if answer else ""
        return float(1.0 if is_solved else 0.0), {"is_solved": is_solved, "pred": pred, "gold": gold}


class MmluAdapter(SystemRouterDataset):
    dataset_name = "mmlu"
    role_domain = "Commonsense"
    prompt_file = os.path.join("MAR", "Roles", "FinalNode", "mmlu.json")

    def __init__(
        self,
        split: str,
        *,
        dataset_root: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.dataset_root = dataset_root
        super().__init__(split, seed=seed)

    def _resolve_split(self) -> str:
        if self.split in ("train", "dev"):
            return "dev"
        if self.split == "val":
            return "val"
        return "test"

    def _load_samples(self) -> List[SystemRouterSample]:
        split = self._resolve_split()
        dataset = MMLUDataset(split, data_root=self.dataset_root)
        samples: List[SystemRouterSample] = []
        for idx in range(len(dataset)):
            record = dataset[idx]
            query = dataset.record_to_input(record)["task"]
            answer = dataset.record_to_target_answer(record)
            item_id = _resolve_item_id(record, idx)
            samples.append(SystemRouterSample(query=query, tests=None, answer=answer, item_id=item_id))
        return samples

    def score_response(
        self,
        response: str,
        tests: Optional[List[str]],
        sample: Optional[SystemRouterSample],
    ) -> Tuple[float, Dict[str, object]]:
        answer = sample.answer if sample else ""
        pred = _postprocess_mmlu_answer(response)
        is_solved = str(pred).strip().upper() == str(answer).strip().upper()
        return float(1.0 if is_solved else 0.0), {"is_solved": is_solved, "pred": pred, "gold": answer}


def get_dataset_adapter(
    dataset_name: str,
    *,
    split: str,
    seed: int = 42,
    dataset_path: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    split_ratio: float = 0.2,
) -> SystemRouterDataset:
    name = dataset_name.strip().lower()
    if name == "mbpp":
        return MbppAdapter(split, seed=seed)
    if name == "humaneval":
        return HumanEvalAdapter(
            split,
            dataset_path=dataset_path,
            split_ratio=split_ratio,
            seed=seed,
        )
    if name == "gsm8k":
        return Gsm8kAdapter(
            split,
            dataset_path=dataset_path,
            train_path=train_path,
            test_path=test_path,
            split_ratio=split_ratio,
            seed=seed,
        )
    if name == "math":
        return MathAdapter(
            split,
            dataset_root=dataset_root or dataset_path,
            seed=seed,
        )
    if name == "mmlu":
        return MmluAdapter(
            split,
            dataset_root=dataset_root or dataset_path,
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def available_datasets() -> Sequence[str]:
    return ("mbpp", "humaneval", "gsm8k", "math", "mmlu")
