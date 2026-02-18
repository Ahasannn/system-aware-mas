#!/usr/bin/env bash
set -euo pipefail

# serve_full_pool.sh
# Serves all models defined in llm_profile_full.json using vLLM

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ==============================================================================
# STORAGE CONFIGURATION
# ==============================================================================
# Load centralized HPC environment configuration
if [[ -f "${ROOT_DIR}/scripts/setup_hpc_env.sh" ]]; then
    source "${ROOT_DIR}/scripts/setup_hpc_env.sh"
else
    echo "[ERROR] setup_hpc_env.sh not found at ${ROOT_DIR}/scripts/setup_hpc_env.sh"
    exit 1
fi

# LOGS -> LOCAL PROJECT FOLDER (job-specific when running under SLURM)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Running under SLURM - use job-specific subdirectory to avoid conflicts
    LOG_DIR="${ROOT_DIR}/logs/vllm/job_${SLURM_JOB_ID}"
    echo "[Setup] SLURM Job ID:            ${SLURM_JOB_ID} (isolated logs/PIDs)"
else
    # Interactive/local run - use shared directory
    LOG_DIR="${ROOT_DIR}/logs/vllm/job_local"
fi
mkdir -p "${LOG_DIR}"

echo "[Setup] HF Cache (Weights):      ${HF_HOME}"
echo "[Setup] HF Token:                $(if [[ -n "${HF_TOKEN}" ]]; then echo "✓ Set"; else echo "✗ Not found"; fi)"
echo "[Setup] Torch Cache:             ${TORCH_HOME}"
echo "[Setup] Triton Cache:            ${TRITON_CACHE_DIR}"
echo "[Setup] TorchInductor Cache:     ${TORCHINDUCTOR_CACHE_DIR}"
echo "[Setup] TMPDIR:                  ${TMPDIR}"
echo "[Setup] Logs (Text):             ${LOG_DIR}"
# ==============================================================================

# ==============================================================================
# CUDA CONFIGURATION (HPC Cluster)
# ==============================================================================
# Load CUDA module if not already loaded (required on HPC clusters)
if ! command -v nvcc &> /dev/null; then
  echo "[Setup] Loading CUDA module..."
  module load cuda/12.8.1
  echo "[Setup] CUDA loaded: $(nvcc --version | head -1)"
else
  echo "[Setup] CUDA already available: $(nvcc --version | head -1)"
fi
# ==============================================================================

# LLM Profile JSON - the single source of truth for model configurations
LLM_PROFILE_JSON="${LLM_PROFILE_JSON:-${ROOT_DIR}/MAR/LLM/llm_profile_full.json}"

echo "[DEBUG] Checking LLM profile JSON..."
if [[ ! -f "${LLM_PROFILE_JSON}" ]]; then
  echo "[vLLM] Error: LLM profile not found: ${LLM_PROFILE_JSON}"
  exit 1
fi
echo "[DEBUG] LLM profile found: ${LLM_PROFILE_JSON}"

# Network binding
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# Optional server-side API key enforcement (leave empty to disable)
VLLM_API_KEY="${VLLM_API_KEY:-}"

# Python interpreter
VLLM_PYTHON="${VLLM_PYTHON:-}"
if [[ -z "${VLLM_PYTHON}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    VLLM_PYTHON="${ROOT_DIR}/.venv/bin/python"
    export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
  else
    # Prevent crash if python isn't immediately found (allows explicit check later)
    VLLM_PYTHON="$(command -v python || true)"
  fi
fi

if [[ -z "${VLLM_PYTHON}" ]]; then
  echo "Error: Python not found! Please activate your venv or 'module load python'."
  exit 1
fi

echo "[DEBUG] Python interpreter: ${VLLM_PYTHON}"
echo "[DEBUG] Checking 'import vllm' ..."
if ! "${VLLM_PYTHON}" -c "import vllm; print('vllm ok')" 2>&1; then
  echo "[vLLM] Missing vllm in ${VLLM_PYTHON}."
  echo "       Run: uv sync --frozen --extra serve"
  exit 1
fi
echo "[DEBUG] vllm import check passed"

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

# Detect if vLLM crashed during startup by checking logs for fatal errors
detect_startup_failure() {
  local logfile="$1"

  if [[ ! -f "${logfile}" ]]; then
    return 1  # No log file, can't detect
  fi

  # Check last 100 lines for fatal initialization errors
  if tail -100 "${logfile}" 2>/dev/null | grep -qE "(ERROR.*EngineCore failed|ValueError.*cache blocks|RuntimeError.*initialization failed|Available KV cache memory: -|EngineCore failed to start)"; then
    return 0  # Crash detected
  fi

  return 1  # No crash detected
}

# Wait for a vLLM server to become healthy with crash detection
wait_for_health() {
  local name="$1"
  local port="$2"
  local pidfile="${LOG_DIR}/${name}.pid"
  local logfile="${LOG_DIR}/${name}.log"

  local timeout_s="${VLLM_STARTUP_TIMEOUT_SECONDS:-7200}"
  local start_s
  start_s="$(date +%s)"

  while true; do
    # Check if process died
    if [[ -f "${pidfile}" ]] && ! kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
      # Process is dead - check if it was a crash during initialization
      if detect_startup_failure "${logfile}"; then
        echo "[vLLM] ${name} crashed during initialization (see ${logfile})"
        return 2  # Special code: needs retry
      else
        echo "[vLLM] ${name} failed to start (see ${logfile})"
        return 1  # Permanent failure
      fi
    fi

    # Check if healthy
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[vLLM] ${name} healthy"
      return 0  # Success
    fi

    # Check timeout
    if (( $(date +%s) - start_s > timeout_s )); then
      echo "[vLLM] ${name} did not become healthy within ${timeout_s}s (see ${logfile})"
      return 1  # Timeout failure
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
  echo "  Log File: ${logfile}"

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

  # Note: HF_HOME environment variable handles the model weight location automatically here
  # --no-enable-prefix-caching: Disable prefix caching to reduce memory overhead
  # --scheduling-policy priority: Enable priority-based queue scheduling (EDF via InfraMind)
  #   Lower priority value = served first. Default 0 when not set (FCFS-equivalent).
  #   InfraMind sets priority = int(deadline) where deadline = arrival_time + budget.
  # VLLM_USE_V1=0: Force v0 engine for stability under high concurrent load
  #   v1 engine (default in 0.14.0) crashes with EngineDeadError / AssertionError
  #   in is_strictly_contiguous(decode_query) when batching many concurrent requests
  VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES="${gpu_device}" nohup "${VLLM_ENTRYPOINT[@]}" \
    --host "${VLLM_HOST}" \
    --port "${port}" \
    --model "${model_name}" \
    --served-model-name "${model_name}" \
    --dtype "${dtype}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --tensor-parallel-size "${tensor_parallel_size}" \
    --no-enable-prefix-caching \
    --scheduling-policy priority \
    --max-num-seqs 32 \
    --swap-space 16 \
    "${extra_flags[@]}" \
    >"${logfile}" 2>&1 &

  echo "$!" > "${pidfile}"
  echo "[vLLM] ${server_name} pid $(cat "${pidfile}") (log: ${logfile})"

  # Return port for health check
  echo "${port}" > "${LOG_DIR}/${server_name}.port"
}

# Main: Start all models from JSON
echo "[DEBUG] About to load model configurations..."
echo "[vLLM] Loading model configurations from: ${LLM_PROFILE_JSON}"
echo ""

echo "[DEBUG] Calling get_model_count()..."
MODEL_COUNT="$(get_model_count)"
echo "[DEBUG] get_model_count() returned: ${MODEL_COUNT}"
echo "[vLLM] Found ${MODEL_COUNT} models to serve"
echo ""

# Start each model server with retry logic
MAX_RETRIES=3

for (( i=0; i<MODEL_COUNT; i++ )); do
  model_name="$(get_model_field "${i}" "Name")"
  server_name="$(echo "${model_name}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
  port="$(get_model_field "${i}" "port")"

  retry_count=0
  success=0

  while (( retry_count < MAX_RETRIES )); do
    # Start the server (skip if already running)
    start_server_from_json "${i}"

    # Wait for health check
    wait_for_health "${server_name}" "${port}"
    health_status=$?

    if [[ ${health_status} -eq 0 ]]; then
      # Success!
      success=1
      break
    elif [[ ${health_status} -eq 2 ]]; then
      # Crash detected during initialization - retry
      retry_count=$((retry_count + 1))
      if (( retry_count < MAX_RETRIES )); then
        wait_time=$((retry_count * 5))
        echo "[vLLM] ${server_name} retry ${retry_count}/${MAX_RETRIES} - waiting ${wait_time}s for GPU memory cleanup..."

        # Kill the crashed process if still alive
        pidfile="${LOG_DIR}/${server_name}.pid"
        if [[ -f "${pidfile}" ]]; then
          pid=$(cat "${pidfile}")
          kill -9 "${pid}" 2>/dev/null || true
          rm -f "${pidfile}"
        fi

        # Wait for GPU memory to be released naturally
        sleep "${wait_time}"

        echo "[vLLM] Retrying ${server_name}..."
      fi
    else
      # Permanent failure (status 1)
      echo "[vLLM] ERROR: ${server_name} failed permanently"
      exit 1
    fi
  done

  if [[ ${success} -eq 0 ]]; then
    echo "[vLLM] ERROR: ${server_name} failed after ${MAX_RETRIES} retries"
    echo "[vLLM] Log: ${LOG_DIR}/${server_name}.log"
    exit 1
  fi

  echo ""
done

echo ""
echo "[MasRouter] All ${MODEL_COUNT} models are now serving!"
echo ""
echo "[MasRouter] Model URLs are read from: ${LLM_PROFILE_JSON}"
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