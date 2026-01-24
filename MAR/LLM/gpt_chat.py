import aiohttp
import asyncio
import atexit
import threading
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json
import requests
from pathlib import Path
from functools import lru_cache
from groq import Groq, AsyncGroq
from openai import OpenAI, AsyncOpenAI

try:
    from openai import APITimeoutError
except Exception:  # pragma: no cover - optional import
    APITimeoutError = None

try:
    import httpx
except Exception:  # pragma: no cover - optional import
    httpx = None

from MAR.LLM.price import cost_count
from MAR.LLM.llm import LLM
from MAR.LLM.llm_registry import LLMRegistry

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')

_SYNC_CLIENT_LOCK = threading.Lock()
_SYNC_CLIENTS: Dict[tuple[str, str], OpenAI] = {}


def _get_shared_sync_client(*, base_url: Optional[str], api_key: str) -> OpenAI:
    # Keep one OpenAI sync client per (base_url, api_key) to reuse HTTP connection pooling.
    key = ((base_url or "").strip(), (api_key or "").strip())
    with _SYNC_CLIENT_LOCK:
        client = _SYNC_CLIENTS.get(key)
        if client is None:
            client = OpenAI(base_url=base_url, api_key=api_key)
            _SYNC_CLIENTS[key] = client
    return client


@atexit.register
def _close_shared_sync_clients() -> None:
    # Best-effort cleanup; prevents dangling sockets/FDS on long runs.
    with _SYNC_CLIENT_LOCK:
        clients = list(_SYNC_CLIENTS.values())
        _SYNC_CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

@lru_cache(maxsize=1)
def _get_test_config() -> Dict[str, Any]:
    """
    Optional local test config file to avoid manual `export ...` steps.

    If `MODEL_BASE_URLS` is not set, we look for:
    - `<repo>/config_test.json`
    - `<repo>/logs/vllm/model_base_urls.json` (written by `scripts/vllm/serve_pool.sh`)
    """
    candidates = [
        _project_root() / "config_test.json",
        _project_root() / "MAR" / "LLM" / "llm_profile_full.json",
        _project_root() / "logs" / "vllm" / "model_base_urls.json",
    ]

    for path in candidates:
        try:
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except OSError:
            continue
        except json.JSONDecodeError:
            continue
    return {}

def _extract_model_base_urls(data: Any) -> Dict[str, str]:
    if not isinstance(data, dict):
        return {}

    # Common shapes:
    # - {"model_base_urls": {...}, "key": "..."}
    # - {"MODEL_BASE_URLS": {...}, "KEY": "..."}
    # - {"llm_pool": {"model_base_urls": {...}}}
    for top_key in ("model_base_urls", "MODEL_BASE_URLS"):
        value = data.get(top_key)
        if isinstance(value, dict):
            data = value
            break
    else:
        pool = data.get("llm_pool")
        if isinstance(pool, dict):
            for pool_key in ("model_base_urls", "MODEL_BASE_URLS"):
                value = pool.get(pool_key)
                if isinstance(value, dict):
                    data = value
                    break

    normalized: Dict[str, str] = {}
    if not isinstance(data, dict):
        return {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized

def _resolve_api_key() -> str:
    key = os.environ.get("KEY")
    if key:
        return key

    config = _get_test_config()
    for key_name in ("key", "KEY", "api_key", "API_KEY"):
        value = config.get(key_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    pool = config.get("llm_pool")
    if isinstance(pool, dict):
        for key_name in ("key", "KEY", "api_key", "API_KEY"):
            value = pool.get(key_name)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return "EMPTY"

@lru_cache(maxsize=1)
def _get_model_base_urls() -> Dict[str, str]:
    """
    Optional per-model base URL mapping for OpenAI-compatible backends.

    Use cases:
    - Run multiple vLLM OpenAI servers on different ports/GPUs.
    - Keep a single client interface while routing by model name.

    Configure via:
    - `MODEL_BASE_URLS` as a JSON object string; or
    - `MODEL_BASE_URLS` as a path to a JSON file on disk.

    Example (env var):
      MODEL_BASE_URLS='{"mistralai/Mistral-7B-Instruct-v0.3":"http://localhost:8003/v1"}'
    """
    raw = os.environ.get("MODEL_BASE_URLS", "").strip()
    if raw:
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return {}
        else:
            try:
                with open(raw, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except OSError:
                return {}
            except json.JSONDecodeError:
                return {}
        return _extract_model_base_urls(data)

    return _extract_model_base_urls(_get_test_config())

def _resolve_base_url(model_name: str) -> Optional[str]:
    per_model = _get_model_base_urls()
    if model_name in per_model:
        return per_model[model_name]
    return os.environ.get("URL")

def _default_request_timeout() -> Optional[float]:
    raw = os.environ.get("LLM_REQUEST_TIMEOUT", "").strip()
    if not raw:
        return 120.0
    try:
        timeout = float(raw)
    except ValueError:
        return 120.0
    if timeout <= 0:
        return None
    return timeout


def _normalize_request_timeout(request_timeout: Optional[float]) -> Optional[float]:
    if request_timeout is None:
        return _default_request_timeout()
    try:
        timeout = float(request_timeout)
    except (TypeError, ValueError):
        return _default_request_timeout()
    if timeout <= 0:
        return None
    return timeout


_TIMEOUT_EXCEPTIONS = [TimeoutError, asyncio.TimeoutError, requests.exceptions.Timeout]
if APITimeoutError is not None:
    _TIMEOUT_EXCEPTIONS.append(APITimeoutError)
if httpx is not None:
    _TIMEOUT_EXCEPTIONS.append(httpx.TimeoutException)
_TIMEOUT_EXCEPTIONS = tuple(_TIMEOUT_EXCEPTIONS)


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, _TIMEOUT_EXCEPTIONS):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


@LLMRegistry.register('ALLChat')
class ALLChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(
        wait=wait_random_exponential(max=100),
        stop=stop_after_attempt(10),
        retry=retry_if_not_exception_type(_TIMEOUT_EXCEPTIONS),
    )
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        timeout = _normalize_request_timeout(request_timeout)
        base_client = None
        try:
            base_client = _get_shared_sync_client(
                base_url=_resolve_base_url(self.model_name),
                api_key=_resolve_api_key(),
            )
            client = base_client.with_options(timeout=timeout)
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_comps,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                raise TimeoutError("LLM request timed out") from exc
            raise
        response = chat_completion.choices[0].message.content
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response, self.model_name)
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        timeout = _normalize_request_timeout(request_timeout)
        client = None
        try:
            client = AsyncOpenAI(
                base_url=_resolve_base_url(self.model_name),
                api_key=_resolve_api_key(),
                timeout=timeout,
            )
            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                raise TimeoutError("LLM request timed out") from exc
            raise
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
        response = chat_completion.choices[0].message.content

        return response
    

@LLMRegistry.register('Deepseek')
class DSChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(
        wait=wait_random_exponential(max=100),
        stop=stop_after_attempt(10),
        retry=retry_if_not_exception_type(_TIMEOUT_EXCEPTIONS),
    )
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        timeout = _normalize_request_timeout(request_timeout)
        base_client = None
        try:
            base_client = _get_shared_sync_client(
                base_url=os.environ.get("DS_URL"),
                api_key=os.environ.get("DS_KEY") or "",
            )
            client = base_client.with_options(timeout=timeout)
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_comps,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                raise TimeoutError("LLM request timed out") from exc
            raise
        response = chat_completion.choices[0].message.content
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response, self.model_name)
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        timeout = _normalize_request_timeout(request_timeout)
        client = None
        try:
            client = AsyncOpenAI(
                base_url=os.environ.get("DS_URL"),
                api_key=os.environ.get("DS_KEY"),
                timeout=timeout,
            )
            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                raise TimeoutError("LLM request timed out") from exc
            raise
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
        response = chat_completion.choices[0].message.content

        return response

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def achat(
    model: str,
    msg: List[Dict],
    request_timeout: Optional[float] = None,
    ):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model + '-y',
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    timeout = _normalize_request_timeout(request_timeout)
    client_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            response_data = await response.json()
            if isinstance(response_data['data'],str):
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt,response_data['data'], model)
                return response_data['data']
            else:
                raise Exception("api error")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))   
def chat(
    model: str,
    msg: List[Dict],
    request_timeout: Optional[float] = None,
    ):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model+'-y',
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    timeout = _normalize_request_timeout(request_timeout)
    response = requests.post(request_url, headers=headers, json=data, timeout=timeout)
    response_data = response.json()
    if isinstance(response_data['data'],str):
        prompt = "".join([item['content'] for item in msg])
        cost_count(prompt,response_data['data'], model)
        return response_data['data']
    else:
        raise Exception("api error")

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        return await achat(self.model_name, messages, request_timeout=request_timeout)
    
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        return chat(self.model_name, messages, request_timeout=request_timeout)
    

@LLMRegistry.register('Groq')
class GroqChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:
        # TODO: Add num_comps to the request
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
        chat_completion = client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        )
        response = chat_completion.choices[0].message.content
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response, self.model_name)        
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:
        # TODO: Add num_comps to the request
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"),)
        chat_completion = await client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        max_tokens = max_tokens,
        temperature = temperature,
        )
        response = chat_completion.choices[0].message.content

        return response
    

@LLMRegistry.register('OpenRouter')
class OpenRouterChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        request_timeout: Optional[float] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        timeout = _normalize_request_timeout(request_timeout)
        client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            timeout=timeout,
        )
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        response = chat_completion.choices[0].message.content
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        # TODO
        return 0
