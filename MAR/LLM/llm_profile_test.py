import json
from pathlib import Path
from typing import Dict, List

_PROFILE_PATH = Path(__file__).with_suffix(".json")
_PROFILE_DATA = {}
if _PROFILE_PATH.is_file():
    try:
        with _PROFILE_PATH.open("r", encoding="utf-8") as file:
            _PROFILE_DATA = json.load(file)
    except (OSError, json.JSONDecodeError):
        _PROFILE_DATA = {}

llm_profile: List[Dict[str, str]] = _PROFILE_DATA.get("models", [])

DEFAULT_MAX_MODEL_LEN: int = _PROFILE_DATA.get("default_max_model_len", 4096)
_MAX_MODEL_LEN_MAP = {
    entry.get("Name"): entry.get("MaxModelLen", DEFAULT_MAX_MODEL_LEN)
    for entry in llm_profile
    if isinstance(entry, dict)
}

DEFAULT_MAX_OUTPUT_TOKENS: int = _PROFILE_DATA.get("default_max_output_tokens", 512)
_MAX_OUTPUT_TOKENS_MAP = {
    entry.get("Name"): entry.get("MaxOutputTokens", DEFAULT_MAX_OUTPUT_TOKENS)
    for entry in llm_profile
    if isinstance(entry, dict)
}

def get_model_max_context_len(model_name: str) -> int:
    return _MAX_MODEL_LEN_MAP.get(model_name, DEFAULT_MAX_MODEL_LEN)

def get_model_max_output_tokens(model_name: str) -> int:
    return _MAX_OUTPUT_TOKENS_MAP.get(model_name, DEFAULT_MAX_OUTPUT_TOKENS)
