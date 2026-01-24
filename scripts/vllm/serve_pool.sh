#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/vllm"
mkdir -p "${LOG_DIR}"

# GPU assignment (for a 2x GPU workstation)
# - Two models run on GPU0
# - One model runs on GPU1
#
# Backwards compatible env vars:
# - `VLLM_SMALL_GPU` == GPU0
# - `VLLM_BIG_GPU`   == GPU1
VLLM_GPU0="${VLLM_GPU0:-${VLLM_SMALL_GPU:-0}}"
VLLM_GPU1="${VLLM_GPU1:-${VLLM_BIG_GPU:-1}}"

# Ports
VLLM_QWEN_PORT="${VLLM_QWEN_PORT:-8001}"
VLLM_LLAMA_PORT="${VLLM_LLAMA_PORT:-8002}"
VLLM_MISTRAL_PORT="${VLLM_MISTRAL_PORT:-8003}"

# Memory / context limits (tune for your GPUs)
# Backwards compatible env vars:
# - `VLLM_SMALL_GPU_MEMORY_UTILIZATION` == GPU0
# - `VLLM_BIG_GPU_MEMORY_UTILIZATION`   == GPU1
#
# Important: vLLM assumes it "owns" the whole GPU when sizing KV cache.
# If you run 2 vLLM servers on the same GPU, keep each utilization low so the
# *sum* stays comfortably below 1.0 (e.g., 0.40 + 0.40).
VLLM_GPU0_MEMORY_UTILIZATION="${VLLM_GPU0_MEMORY_UTILIZATION:-${VLLM_SMALL_GPU_MEMORY_UTILIZATION:-0.40}}"
VLLM_GPU1_MEMORY_UTILIZATION="${VLLM_GPU1_MEMORY_UTILIZATION:-${VLLM_BIG_GPU_MEMORY_UTILIZATION:-0.90}}"
#
# Note: running 2 servers on the same GPU usually requires a smaller KV cache.
# For most evals (MBPP/HumanEval/GSM8K/etc), 4096 is plenty.
VLLM_SMALL_MAX_MODEL_LEN="${VLLM_SMALL_MAX_MODEL_LEN:-4096}"
VLLM_BIG_MAX_MODEL_LEN="${VLLM_BIG_MAX_MODEL_LEN:-8192}"

LLM_PROFILE_JSON="${LLM_PROFILE_JSON:-${ROOT_DIR}/MAR/LLM/llm_profile_full.json}"
VLLM_LLAMA_MAX_LEN="${VLLM_LLAMA_MAX_LEN:-}"
VLLM_QWEN_MAX_LEN="${VLLM_QWEN_MAX_LEN:-}"
VLLM_MISTRAL_MAX_LEN="${VLLM_MISTRAL_MAX_LEN:-}"

_get_model_max_len() {
  local model_name="$1"
  python - <<PY "$model_name"
import json
from pathlib import Path
import sys

path = Path("$LLM_PROFILE_JSON")
default_len = 4096
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    default_len = int(data.get("default_max_model_len", default_len))
except Exception:
    print(default_len)
    sys.exit(0)

for entry in data.get("models", []):
    if isinstance(entry, dict) and entry.get("Name") == model_name:
        print(entry.get("MaxModelLen", entry.get("max_model_len", default_len)))
        sys.exit(0)
print(default_len)
PY
}

LLAMA_MAX_LEN="${VLLM_LLAMA_MAX_LEN:-$(_get_model_max_len "meta-llama/Llama-3.2-3B-Instruct")}"
QWEN_MAX_LEN="${VLLM_QWEN_MAX_LEN:-$(_get_model_max_len "Qwen/Qwen2.5-3B-Instruct")}"
MISTRAL_MAX_LEN="${VLLM_MISTRAL_MAX_LEN:-$(_get_model_max_len "mistralai/Mistral-7B-Instruct-v0.3")}"

# Per-model overrides (optional).
# If you set these, ensure GPU0 values sum to <= ~0.85 for headroom.
VLLM_LLAMA_GPU_MEMORY_UTILIZATION="${VLLM_LLAMA_GPU_MEMORY_UTILIZATION:-${VLLM_GPU0_MEMORY_UTILIZATION}}"
VLLM_QWEN_GPU_MEMORY_UTILIZATION="${VLLM_QWEN_GPU_MEMORY_UTILIZATION:-${VLLM_GPU0_MEMORY_UTILIZATION}}"
VLLM_MISTRAL_GPU_MEMORY_UTILIZATION="${VLLM_MISTRAL_GPU_MEMORY_UTILIZATION:-${VLLM_GPU1_MEMORY_UTILIZATION}}"

# Network binding
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# Optional server-side API key enforcement (leave empty to disable).
# If you set this, also set `KEY` to the same value for the client.
VLLM_API_KEY="${VLLM_API_KEY:-}"

# Some models may require remote code; default is off for safety.
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"

# Read model URLs from the LLM profile (model_base_urls section)
_get_model_url() {
  local model_name="$1"
  python - <<PY "$model_name"
import json
from pathlib import Path
import sys

path = Path("$LLM_PROFILE_JSON")
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    urls = data.get("model_base_urls", {})
    url = urls.get(sys.argv[1], "")
    print(url)
except Exception:
    print("")
PY
}

_get_port_from_url() {
  local url="$1"
  echo "$url" | sed -n 's|.*:\([0-9]*\)/.*|\1|p'
}

# Legacy: also write to model_base_urls.json for backward compatibility
MODEL_BASE_URLS_FILE="${MODEL_BASE_URLS_FILE:-${LOG_DIR}/model_base_urls.json}"

cat >"${MODEL_BASE_URLS_FILE}" <<EOF
{
  "Qwen/Qwen2.5-3B-Instruct": "http://127.0.0.1:${VLLM_QWEN_PORT}/v1",
  "meta-llama/Llama-3.2-3B-Instruct": "http://127.0.0.1:${VLLM_LLAMA_PORT}/v1",
  "mistralai/Mistral-7B-Instruct-v0.3": "http://127.0.0.1:${VLLM_MISTRAL_PORT}/v1"
}
EOF

API_KEY_FLAGS=()
if [[ -n "${VLLM_API_KEY}" ]]; then
  API_KEY_FLAGS=(--api-key "${VLLM_API_KEY}")
fi

TRUST_REMOTE_CODE_FLAGS=()
if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_REMOTE_CODE_FLAGS=(--trust-remote-code)
fi

VLLM_PYTHON="${VLLM_PYTHON:-}"
if [[ -z "${VLLM_PYTHON}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    VLLM_PYTHON="${ROOT_DIR}/.venv/bin/python"
  else
    VLLM_PYTHON="$(command -v python)"
  fi
fi

if ! "${VLLM_PYTHON}" -c "import vllm, triton, setuptools" >/dev/null 2>&1; then
  echo "[vLLM] Missing runtime deps in ${VLLM_PYTHON}."
  echo "       Run: uv sync --frozen --extra serve"
  exit 1
fi

VLLM_ENTRYPOINT=("${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server)

wait_for_health() {
  local name="$1"
  local port="$2"
  local pidfile="${LOG_DIR}/${name}.pid"
  local logfile="${LOG_DIR}/${name}.log"

  local timeout_s="${VLLM_STARTUP_TIMEOUT_SECONDS:-300}"
  local start_s
  start_s="$(date +%s)"

  while true; do
    if [[ -f "${pidfile}" ]] && ! kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
      echo "[vLLM] ${name} failed to start (see ${logfile})"
      return 1
    fi

    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[vLLM] ${name} healthy"
      return 0
    fi

    if (( $(date +%s) - start_s > timeout_s )); then
      echo "[vLLM] ${name} did not become healthy within ${timeout_s}s (see ${logfile})"
      return 1
    fi

    sleep 2
  done
}

start_server() {
  local name="$1"
  local gpu="$2"
  local model="$3"
  local port="$4"
  local gpu_mem_util="$5"
  local max_model_len="$6"
  local tokenizer_mode="${7:-auto}"

  local logfile="${LOG_DIR}/${name}.log"
  local pidfile="${LOG_DIR}/${name}.pid"

  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "[vLLM] ${name} already running (pid $(cat "${pidfile}"))"
    return 0
  fi

  echo "[vLLM] Starting ${name}"
  echo "  GPU:   ${gpu}"
  echo "  Port:  ${port}"
  echo "  Model: ${model}"

  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM_ENTRYPOINT[@]}" \
    --host "${VLLM_HOST}" \
    --port "${port}" \
    --model "${model}" \
    --served-model-name "${model}" \
    --dtype auto \
    --gpu-memory-utilization "${gpu_mem_util}" \
    --max-model-len "${max_model_len}" \
    --tokenizer-mode "${tokenizer_mode}" \
    "${API_KEY_FLAGS[@]}" \
    "${TRUST_REMOTE_CODE_FLAGS[@]}" \
    >"${logfile}" 2>&1 &

  echo "$!" > "${pidfile}"
  echo "[vLLM] ${name} pid $(cat "${pidfile}") (log: ${logfile})"
}

start_server "llama_3b" "${VLLM_GPU0}" "meta-llama/Llama-3.2-3B-Instruct" "${VLLM_LLAMA_PORT}" "${VLLM_LLAMA_GPU_MEMORY_UTILIZATION}" "${LLAMA_MAX_LEN}"
wait_for_health "llama_3b" "${VLLM_LLAMA_PORT}"

start_server "qwen_3b" "${VLLM_GPU0}" "Qwen/Qwen2.5-3B-Instruct" "${VLLM_QWEN_PORT}" "${VLLM_QWEN_GPU_MEMORY_UTILIZATION}" "${QWEN_MAX_LEN}"
wait_for_health "qwen_3b" "${VLLM_QWEN_PORT}"

start_server "mistral_7b" "${VLLM_GPU1}" "mistralai/Mistral-7B-Instruct-v0.3" "${VLLM_MISTRAL_PORT}" "${VLLM_MISTRAL_GPU_MEMORY_UTILIZATION}" "${MISTRAL_MAX_LEN}" "mistral"
wait_for_health "mistral_7b" "${VLLM_MISTRAL_PORT}"

echo ""
echo "[MasRouter] Point the client at the per-model endpoints:"
echo "  export MODEL_BASE_URLS=\"${MODEL_BASE_URLS_FILE}\""
if [[ -n "${VLLM_API_KEY}" ]]; then
  echo "  export KEY=\"${VLLM_API_KEY}\""
else
  echo "  export KEY=\"EMPTY\""
fi
echo ""
echo "[Quick check] (optional)"
echo "  curl -s \"http://127.0.0.1:${VLLM_QWEN_PORT}/v1/models\" | head"
echo ""
echo "[Stop]"
echo "  bash \"${ROOT_DIR}/scripts/vllm/stop_pool.sh\""
