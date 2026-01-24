#!/usr/bin/env bash
set -euo pipefail

# serve_full_pool.sh
# Serves all models defined in llm_profile_full.json using vLLM

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/vllm"
mkdir -p "${LOG_DIR}"

# LLM Profile JSON - the single source of truth for model configurations
LLM_PROFILE_JSON="${LLM_PROFILE_JSON:-${ROOT_DIR}/MAR/LLM/llm_profile_full.json}"

if [[ ! -f "${LLM_PROFILE_JSON}" ]]; then
  echo "[vLLM] Error: LLM profile not found: ${LLM_PROFILE_JSON}"
  exit 1
fi

# Network binding
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# Optional server-side API key enforcement (leave empty to disable)
VLLM_API_KEY="${VLLM_API_KEY:-}"

# Python interpreter
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

# Helper: Extract model count from JSON
get_model_count() {
  "${VLLM_PYTHON}" - <<PY
import json
from pathlib import Path

data = json.loads(Path("${LLM_PROFILE_JSON}").read_text(encoding="utf-8"))
print(len(data.get("models", [])))
PY
}

# Helper: Extract model config field by index
get_model_field() {
  local index="$1"
  local field="$2"
  "${VLLM_PYTHON}" - <<PY
import json
from pathlib import Path

data = json.loads(Path("${LLM_PROFILE_JSON}").read_text(encoding="utf-8"))
models = data.get("models", [])
if ${index} < len(models):
    model = models[${index}]
    if "${field}" in model:
        print(model["${field}"])
    elif "${field}" in model.get("vllm_config", {}):
        print(model["vllm_config"]["${field}"])
    else:
        print("")
else:
    print("")
PY
}

# Helper: Get default max model len from global settings
get_default_max_model_len() {
  "${VLLM_PYTHON}" - <<PY
import json
from pathlib import Path

data = json.loads(Path("${LLM_PROFILE_JSON}").read_text(encoding="utf-8"))
print(data.get("global_settings", {}).get("default_max_model_len", 16384))
PY
}

# Wait for a vLLM server to become healthy
wait_for_health() {
  local name="$1"
  local port="$2"
  local pidfile="${LOG_DIR}/${name}.pid"
  local logfile="${LOG_DIR}/${name}.log"

  local timeout_s="${VLLM_STARTUP_TIMEOUT_SECONDS:-600}"
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

# Start a vLLM server with config from JSON
start_server_from_json() {
  local index="$1"

  # Extract all config from JSON
  local model_name
  local port
  local gpu_device
  local gpu_memory_utilization
  local max_model_len
  local dtype
  local trust_remote_code
  local enforce_eager
  local tensor_parallel_size

  model_name="$(get_model_field "${index}" "Name")"
  port="$(get_model_field "${index}" "port")"
  gpu_device="$(get_model_field "${index}" "gpu_device")"
  gpu_memory_utilization="$(get_model_field "${index}" "gpu_memory_utilization")"
  max_model_len="$(get_model_field "${index}" "MaxModelLen")"
  dtype="$(get_model_field "${index}" "dtype")"
  trust_remote_code="$(get_model_field "${index}" "trust_remote_code")"
  enforce_eager="$(get_model_field "${index}" "enforce_eager")"
  tensor_parallel_size="$(get_model_field "${index}" "tensor_parallel_size")"

  # Use defaults if not specified
  max_model_len="${max_model_len:-$(get_default_max_model_len)}"
  dtype="${dtype:-bfloat16}"
  tensor_parallel_size="${tensor_parallel_size:-1}"

  # Generate a safe server name from the model name
  local server_name
  server_name="$(echo "${model_name}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"

  local logfile="${LOG_DIR}/${server_name}.log"
  local pidfile="${LOG_DIR}/${server_name}.pid"

  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "[vLLM] ${server_name} already running (pid $(cat "${pidfile}"))"
    return 0
  fi

  echo "[vLLM] Starting ${server_name}"
  echo "  Model: ${model_name}"
  echo "  GPU:   ${gpu_device}"
  echo "  Port:  ${port}"
  echo "  Memory Utilization: ${gpu_memory_utilization}"
  echo "  Max Model Len: ${max_model_len}"
  echo "  Tensor Parallel Size: ${tensor_parallel_size}"

  # Build command flags
  local extra_flags=()

  if [[ -n "${VLLM_API_KEY}" ]]; then
    extra_flags+=(--api-key "${VLLM_API_KEY}")
  fi

  if [[ "${trust_remote_code}" == "true" || "${trust_remote_code}" == "True" ]]; then
    extra_flags+=(--trust-remote-code)
  fi

  if [[ "${enforce_eager}" == "true" || "${enforce_eager}" == "True" ]]; then
    extra_flags+=(--enforce-eager)
  fi

  CUDA_VISIBLE_DEVICES="${gpu_device}" nohup "${VLLM_ENTRYPOINT[@]}" \
    --host "${VLLM_HOST}" \
    --port "${port}" \
    --model "${model_name}" \
    --served-model-name "${model_name}" \
    --dtype "${dtype}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --tensor-parallel-size "${tensor_parallel_size}" \
    "${extra_flags[@]}" \
    >"${logfile}" 2>&1 &

  echo "$!" > "${pidfile}"
  echo "[vLLM] ${server_name} pid $(cat "${pidfile}") (log: ${logfile})"

  # Return port for health check
  echo "${port}" > "${LOG_DIR}/${server_name}.port"
}

# Main: Start all models from JSON
echo "[vLLM] Loading model configurations from: ${LLM_PROFILE_JSON}"
echo ""

MODEL_COUNT="$(get_model_count)"
echo "[vLLM] Found ${MODEL_COUNT} models to serve"
echo ""

# Start each model server
for (( i=0; i<MODEL_COUNT; i++ )); do
  start_server_from_json "${i}"

  # Get server name and port for health check
  model_name="$(get_model_field "${i}" "Name")"
  server_name="$(echo "${model_name}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
  port="$(get_model_field "${i}" "port")"

  wait_for_health "${server_name}" "${port}"
  echo ""
done

# Copy model_base_urls from JSON for client reference
MODEL_BASE_URLS_FILE="${MODEL_BASE_URLS_FILE:-${LOG_DIR}/model_base_urls.json}"
"${VLLM_PYTHON}" - <<PY
import json
from pathlib import Path

data = json.loads(Path("${LLM_PROFILE_JSON}").read_text(encoding="utf-8"))
urls = data.get("model_base_urls", {})
Path("${MODEL_BASE_URLS_FILE}").write_text(json.dumps(urls, indent=2) + "\n", encoding="utf-8")
PY

echo ""
echo "[MasRouter] All ${MODEL_COUNT} models are now serving!"
echo ""
echo "[MasRouter] Point the client at the per-model endpoints:"
echo "  export MODEL_BASE_URLS=\"${MODEL_BASE_URLS_FILE}\""
echo "  # Or use the profile directly:"
echo "  export LLM_PROFILE_JSON=\"${LLM_PROFILE_JSON}\""
if [[ -n "${VLLM_API_KEY}" ]]; then
  echo "  export KEY=\"${VLLM_API_KEY}\""
else
  echo "  export KEY=\"EMPTY\""
fi
echo ""
echo "[Model endpoints]"
"${VLLM_PYTHON}" - <<PY
import json
from pathlib import Path

data = json.loads(Path("${LLM_PROFILE_JSON}").read_text(encoding="utf-8"))
for name, url in data.get("model_base_urls", {}).items():
    print(f"  {name}: {url}")
PY
echo ""
echo "[Quick check] (optional)"
echo "  curl -s \"http://127.0.0.1:8001/v1/models\" | head"
echo ""
echo "[Stop]"
echo "  bash \"${ROOT_DIR}/scripts/vllm/stop_pool.sh\""
