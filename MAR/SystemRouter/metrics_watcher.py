import threading
import time
from typing import Dict

import requests
from loguru import logger

# Latest metrics snapshot per model.
model_metrics: Dict[str, Dict[str, float]] = {}

# Last cumulative values per model (for delta computation).
_prev_values: Dict[str, Dict[str, float]] = {}


def fetch_vllm_metrics(model_name: str, url: str) -> None:
    """Fetch metrics from one vLLM endpoint and update global storage."""
    try:
        response = requests.get(url, timeout=2)
        if response.status_code != 200:
            logger.trace("[Metrics] {} returned status {}", url, response.status_code)
            return
        lines = response.text.splitlines()
    except Exception as e:
        logger.trace("[Metrics] Failed to fetch {}: {}", url, e)
        return

    curr: Dict[str, float] = {}
    found_any = False
    for line in lines:
        # Handle both vllm: (old format) and vllm_ (Prometheus format)
        if not (line.startswith("vllm:") or line.startswith("vllm_")):
            continue
        found_any = True
        parts = line.split()
        if not parts:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        # Normalize line for matching (replace both : and _ after vllm prefix)
        if "num_requests_running" in line:
            curr["num_requests_running"] = value
        elif "num_requests_waiting" in line:
            curr["num_requests_waiting"] = value
        elif "kv_cache_usage_perc" in line:
            curr["kv_cache_usage_perc"] = value
        elif "time_to_first_token_seconds_sum" in line:
            curr["ttft_sum"] = value
        elif "time_to_first_token_seconds_count" in line:
            curr["ttft_count"] = value
        elif "inter_token_latency_seconds_sum" in line:
            curr["itl_sum"] = value
        elif "inter_token_latency_seconds_count" in line:
            curr["itl_count"] = value
        elif "e2e_request_latency_seconds_sum" in line:
            curr["e2e_sum"] = value
        elif "e2e_request_latency_seconds_count" in line:
            curr["e2e_count"] = value
        elif "request_queue_time_seconds_sum" in line:
            curr["queue_sum"] = value
        elif "request_queue_time_seconds_count" in line:
            curr["queue_count"] = value
        elif "request_inference_time_seconds_sum" in line:
            curr["inference_sum"] = value
        elif "request_inference_time_seconds_count" in line:
            curr["inference_count"] = value

    if not found_any:
        logger.trace("[Metrics] No vllm metrics found in response from {}", url)

    prev = _prev_values.get(model_name, {})
    prev_snapshot = model_metrics.get(model_name, {})
    data: Dict[str, float] = {}

    for prefix in ("ttft", "itl", "e2e", "queue", "inference"):
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
    logger.trace(
        "[Metrics] {} running={} waiting={} kv_cache={:.2f}% ttft_avg={:.3f}s itl_avg={:.3f}s",
        model_name,
        data.get("num_requests_running", 0),
        data.get("num_requests_waiting", 0),
        data.get("kv_cache_usage_perc", 0) * 100,
        data.get("ttft_avg", 0),
        data.get("itl_avg", 0),
    )


def background_metrics_collector(model_url_map: Dict[str, str], interval: float = 5.0) -> None:
    logger.debug("[Metrics] Background collector started with {} models, interval={}s", len(model_url_map), interval)
    while True:
        for name, url in model_url_map.items():
            fetch_vllm_metrics(name, url)
        time.sleep(interval)


def start_metrics_watcher(model_url_map: Dict[str, str], interval: float = 5.0) -> threading.Thread:
    """Start background metrics watcher thread."""
    logger.info("[Metrics] Starting watcher for {} models", len(model_url_map))
    for name, url in model_url_map.items():
        logger.debug("[Metrics]   {} -> {}", name, url)

    # Initial fetch before starting background thread (ensures metrics available immediately)
    for name, url in model_url_map.items():
        fetch_vllm_metrics(name, url)

    thread = threading.Thread(
        target=background_metrics_collector,
        args=(model_url_map, interval),
        daemon=True,
    )
    thread.start()
    return thread
