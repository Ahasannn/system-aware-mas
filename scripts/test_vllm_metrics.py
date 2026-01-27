#!/usr/bin/env python3
"""Quick test to check if vLLM metrics endpoints are accessible."""

import requests
import json
from pathlib import Path

# Load model URLs from profile
profile_path = Path(__file__).resolve().parents[1] / "MAR" / "LLM" / "llm_profile_full.json"
with open(profile_path) as f:
    config = json.load(f)

model_urls = config.get("model_base_urls", {})

print(f"Testing {len(model_urls)} models from {profile_path}\n")

for model_name, base_url in model_urls.items():
    # Convert base URL to metrics URL
    metrics_url = base_url.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    metrics_url = f"{metrics_url}/metrics"

    print(f"Model: {model_name}")
    print(f"  URL: {metrics_url}")

    try:
        resp = requests.get(metrics_url, timeout=5)
        print(f"  Status: {resp.status_code}")

        if resp.status_code == 200:
            lines = resp.text.splitlines()
            vllm_lines = [l for l in lines if l.startswith("vllm")]
            print(f"  Found {len(vllm_lines)} vllm metric lines")

            # Show metrics we care about - specifically _sum and _count for averages
            print(f"  Looking for _sum and _count metrics:")
            for l in vllm_lines:
                if "_sum" in l or "_count" in l:
                    print(f"    {l}")
        else:
            print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    print()
