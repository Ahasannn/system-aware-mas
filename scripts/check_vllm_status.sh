#!/bin/bash
# Check vLLM server status

echo "=========================================="
echo "vLLM Server Status Check"
echo "=========================================="
echo ""

# Check if on correct node
echo "[1] Current node: $(hostname)"
echo ""

# Check CUDA availability
echo "[2] CUDA Status:"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
else
    echo "✗ nvidia-smi NOT available (not on GPU node)"
fi
echo ""

# Check ports
echo "[3] Port Status:"
for port in 8001 8002 8003 8004 8005; do
    if curl -s -m 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        model=$(curl -s "http://127.0.0.1:${port}/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        echo "  ✓ Port ${port}: HEALTHY (${model})"
    else
        echo "  ✗ Port ${port}: NOT RESPONDING"
    fi
done
echo ""

# Check PID files
echo "[4] vLLM PIDs from logs/vllm/*.pid:"
if ls logs/vllm/*.pid >/dev/null 2>&1; then
    for pidfile in logs/vllm/*.pid; do
        name=$(basename "$pidfile" .pid)
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  ✓ $name (PID $pid): RUNNING"
            else
                echo "  ✗ $name (PID $pid): DEAD"
            fi
        fi
    done
else
    echo "  ✗ No PID files found"
fi
echo ""

# Check vLLM processes
echo "[5] vLLM Processes:"
vllm_procs=$(ps aux | grep -E "vllm.*api_server" | grep -v grep)
if [ -n "$vllm_procs" ]; then
    echo "$vllm_procs" | awk '{printf "  PID %s: %s\n", $2, $NF}'
else
    echo "  ✗ No vLLM API server processes found"
fi
echo ""

# Check llm_profile_full.json
echo "[6] Model Configuration:"
if [ -f "MAR/LLM/llm_profile_full.json" ]; then
    echo "  ✓ MAR/LLM/llm_profile_full.json exists"
    python3 -c "import json; d=json.load(open('MAR/LLM/llm_profile_full.json')); print(json.dumps(d.get('model_base_urls', {}), indent=2))"
else
    echo "  ✗ MAR/LLM/llm_profile_full.json not found"
fi
echo ""

# Recommendation
echo "=========================================="
echo "Recommendation:"
echo "=========================================="
if curl -s -m 2 "http://127.0.0.1:8001/health" >/dev/null 2>&1; then
    echo "✓ vLLM servers are running and healthy!"
    echo "  You can proceed with training/testing."
else
    echo "✗ vLLM servers are NOT running!"
    echo ""
    echo "To start the servers:"
    echo "  bash scripts/vllm/serve_full_pool.sh"
    echo ""
    echo "This will start all 5 models and wait for them to be healthy."
    echo "Expected startup time: 2-5 minutes"
fi
echo ""
