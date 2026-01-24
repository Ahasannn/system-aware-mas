import threading
import time
from typing import Dict

import requests

# Latest metrics snapshot per model.
model_metrics: Dict[str, Dict[str, float]] = {}

# Last cumulative values per model (for delta computation).
_prev_values: Dict[str, Dict[str, float]] = {}


def fetch_vllm_metrics(model_name: str, url: str) -> None:
    """Fetch metrics from one vLLM endpoint and update global storage."""
    try:
        response = requests.get(url, timeout=2)
        if response.status_code != 200:
            return
        lines = response.text.splitlines()
    except Exception:
        return

    curr: Dict[str, float] = {}
    for line in lines:
        if not line.startswith("vllm:"):
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        if line.startswith("vllm:num_requests_running"):
            curr["num_requests_running"] = value
        elif line.startswith("vllm:num_requests_waiting"):
            curr["num_requests_waiting"] = value
        elif line.startswith("vllm:gpu_cache_usage_perc"):
            curr["kv_cache_usage_perc"] = value
        elif line.startswith("vllm:time_to_first_token_seconds_sum"):
            curr["ttft_sum"] = value
        elif line.startswith("vllm:time_to_first_token_seconds_count"):
            curr["ttft_count"] = value
        elif line.startswith("vllm:time_per_output_token_seconds_sum"):
            curr["itl_sum"] = value
        elif line.startswith("vllm:time_per_output_token_seconds_count"):
            curr["itl_count"] = value
        elif line.startswith("vllm:e2e_request_latency_seconds_sum"):
            curr["e2e_sum"] = value
        elif line.startswith("vllm:e2e_request_latency_seconds_count"):
            curr["e2e_count"] = value

    prev = _prev_values.get(model_name, {})
    prev_snapshot = model_metrics.get(model_name, {})
    data: Dict[str, float] = {}

    for prefix in ("ttft", "itl", "e2e"):
        sum_key, cnt_key = f"{prefix}_sum", f"{prefix}_count"
        cur_sum, cur_cnt = curr.get(sum_key, 0.0), curr.get(cnt_key, 0.0)
        prev_sum, prev_cnt = prev.get(sum_key, 0.0), prev.get(cnt_key, 0.0)

        delta_sum = cur_sum - prev_sum
        delta_cnt = cur_cnt - prev_cnt
        if delta_cnt > 0:
            avg = delta_sum / delta_cnt
        else:
            avg = float(prev_snapshot.get(f"{prefix}_avg", 0.0))

        data[f"{prefix}_avg"] = avg
        data[sum_key] = cur_sum
        data[cnt_key] = cur_cnt

    data["num_requests_running"] = curr.get("num_requests_running", 0.0)
    data["num_requests_waiting"] = curr.get("num_requests_waiting", 0.0)
    data["kv_cache_usage_perc"] = curr.get("kv_cache_usage_perc", 0.0)

    _prev_values[model_name] = curr
    model_metrics[model_name] = data


def background_metrics_collector(model_url_map: Dict[str, str], interval: float = 5.0) -> None:
    while True:
        for name, url in model_url_map.items():
            fetch_vllm_metrics(name, url)
        time.sleep(interval)


def start_metrics_watcher(model_url_map: Dict[str, str], interval: float = 5.0) -> threading.Thread:
    """Start background metrics watcher thread."""
    thread = threading.Thread(
        target=background_metrics_collector,
        args=(model_url_map, interval),
        daemon=True,
    )
    thread.start()
    return thread
