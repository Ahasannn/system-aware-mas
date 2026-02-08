"""Per-query latency budget lookup from baseline CSV data."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from loguru import logger


class BudgetProvider:
    """Look up per-query latency budgets from a baseline inference CSV.

    Parses ``record_type="episode"`` rows and builds:
      - ``{(item_id, arrival_rate): workflow_latency_seconds}`` exact lookup
      - ``{arrival_rate: avg_latency}`` per-rate fallback
    """

    def __init__(self, csv_path: Union[str, Path]) -> None:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Budget CSV not found: {path}")

        self._exact: Dict[Tuple[str, float], float] = {}
        rate_accum: Dict[float, list] = defaultdict(list)

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if _safe_str(row.get("record_type")) != "episode":
                    continue
                item_id = _safe_str(row.get("item_id"))
                arrival_rate = _safe_float(row.get("arrival_rate"))
                latency = _safe_float(row.get("workflow_latency_seconds"))
                if not item_id or arrival_rate is None or latency is None:
                    continue
                self._exact[(item_id, arrival_rate)] = latency
                rate_accum[arrival_rate].append(latency)

        self._rate_avg: Dict[float, float] = {}
        for rate, values in rate_accum.items():
            self._rate_avg[rate] = sum(values) / len(values)

        logger.info(
            "[BudgetProvider] Loaded {} exact entries across {} arrival rates from {}",
            len(self._exact),
            len(self._rate_avg),
            path.name,
        )

    @property
    def arrival_rates(self) -> list:
        return sorted(self._rate_avg.keys())

    def get_budget(self, item_id: str, arrival_rate: float) -> float:
        """Return the latency budget for a query at a given arrival rate.

        - Exact match → return the recorded workflow latency.
        - Missing query → return the average latency for that arrival rate
          (with a debug-level warning).
        - Missing arrival rate entirely → ``KeyError``.
        """
        key = (str(item_id), float(arrival_rate))
        exact = self._exact.get(key)
        if exact is not None:
            return exact

        rate = float(arrival_rate)
        if rate not in self._rate_avg:
            raise KeyError(
                f"Arrival rate {rate} not found in budget CSV. "
                f"Available rates: {sorted(self._rate_avg.keys())}"
            )
        avg = self._rate_avg[rate]
        logger.debug(
            "[BudgetProvider] No exact budget for item_id={} rate={}; using rate average {:.3f}s",
            item_id,
            arrival_rate,
            avg,
        )
        return avg


def _safe_str(value: object) -> str:
    if value is None:
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
