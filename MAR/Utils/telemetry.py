from __future__ import annotations

import csv
import json
import threading
import time
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _to_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return value.replace("\n", "\\n")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, dict, tuple)):
        return _json_dumps(value)
    return str(value)

_LLM_USAGE_CONTEXT_KEY: ContextVar[Optional[str]] = ContextVar("llm_usage_context_key", default=None)
_CSV_LOCK = threading.Lock()


class LLMUsageTracker:
    _instance: Optional["LLMUsageTracker"] = None

    @classmethod
    def instance(cls) -> "LLMUsageTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage: Dict[str, Dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0})

    def set_context(self, key: str) -> object:
        return _LLM_USAGE_CONTEXT_KEY.set(key)

    def reset_context(self, token: object) -> None:
        _LLM_USAGE_CONTEXT_KEY.reset(token)

    def clear(self, key: str) -> None:
        with self._lock:
            self._usage.pop(key, None)

    def record(self, *, cost: float, prompt_tokens: int, completion_tokens: int) -> None:
        key = _LLM_USAGE_CONTEXT_KEY.get()
        if not key:
            return
        with self._lock:
            current = self._usage[key]
            current["cost"] += float(cost)
            current["prompt_tokens"] += float(prompt_tokens)
            current["completion_tokens"] += float(completion_tokens)

    def consume(self, key: str) -> Dict[str, float]:
        with self._lock:
            usage = self._usage.pop(key, None)
        if not usage:
            return {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0}
        return usage


@dataclass(frozen=True)
class NodeTiming:
    round_idx: int
    node_id: str
    node_name: str
    role_name: str
    llm_name: str
    is_decision_node: bool
    attempts: int
    success: bool
    error: str
    ts_start: str
    ts_end: str
    duration_sec: float
    cost_delta: float
    prompt_tokens: int
    completion_tokens: int
    output_text: str


class GraphTrace:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workflow_start_perf: Optional[float] = None
        self._workflow_start_ts: Optional[str] = None
        self._workflow_end_perf: Optional[float] = None
        self._workflow_end_ts: Optional[str] = None
        self._workflow_success: Optional[bool] = None
        self._workflow_error: Optional[str] = None
        self.node_events: List[NodeTiming] = []

    def start_workflow(self) -> None:
        with self._lock:
            if self._workflow_start_perf is not None:
                return
            self._workflow_start_perf = time.perf_counter()
            self._workflow_start_ts = utc_now_iso()

    def end_workflow(self, *, success: bool = True, error: str = "") -> None:
        with self._lock:
            if self._workflow_end_perf is not None:
                return
            self._workflow_end_perf = time.perf_counter()
            self._workflow_end_ts = utc_now_iso()
            self._workflow_success = success
            self._workflow_error = error

    def record_node_event(self, event: NodeTiming) -> None:
        with self._lock:
            self.node_events.append(event)

    def workflow_timing(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._workflow_start_perf is None:
                return None
            if self._workflow_end_perf is None:
                return None
            duration_sec = self._workflow_end_perf - self._workflow_start_perf
            return {
                "ts_start": self._workflow_start_ts or "",
                "ts_end": self._workflow_end_ts or "",
                "duration_sec": duration_sec,
                "success": bool(self._workflow_success) if self._workflow_success is not None else True,
                "error": self._workflow_error or "",
            }


DEFAULT_TELEMETRY_FIELDS: Sequence[str] = (
    "run_id",
    "dataset",
    "split",
    "batch_id",
    "item_id",
    "record_type",
    "task_name",
    "reasoning_name",
    "decision_method",
    "graph_id",
    "num_rounds",
    "num_agents",
    "agent_roles_json",
    "agent_llms_json",
    "role_llm_map_json",
    # LLM-specific workflow timing (wall-clock) captured by Graph.
    # Kept separate from `duration_sec`, which is end-to-end workflow wall time.
    "workflow_latency_seconds",
    "llm_elapsed_seconds",
    "node_id",
    "node_name",
    "role_name",
    "llm_name",
    "is_decision_node",
    "round_idx",
    "attempts",
    "success",
    "error",
    "ts_start",
    "ts_end",
    "duration_sec",
    "cost_delta",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "output_text",
    "router_log_prob",
    "router_task_probs_json",
    "router_agent_num_pred",
    "utility",
    "quality_is_correct",
    "quality_pred",
    "quality_gold",
    "quality_feedback",
    "quality_state_json",
    "eval_duration_sec",
    "arrival_rate",
    "arrival_pattern",
)


class CsvTelemetryWriter:
    def __init__(self, path: str | Path, *, fieldnames: Sequence[str] = DEFAULT_TELEMETRY_FIELDS) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._lock = threading.Lock()

    def append_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        with _CSV_LOCK, self._lock:
            file_exists = self.path.exists()
            write_header = (not file_exists) or (self.path.stat().st_size == 0)
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                for row in rows_list:
                    safe_row = {k: _to_csv_value(v) for k, v in row.items()}
                    writer.writerow(safe_row)
