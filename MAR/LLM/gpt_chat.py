import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json
import requests
from pathlib import Path
from functools import lru_cache
from groq import Groq, AsyncGroq
from openai import OpenAI, AsyncOpenAI

from MAR.LLM.price import cost_count
from MAR.LLM.llm import LLM
from MAR.LLM.llm_registry import LLMRegistry

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')

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


@LLMRegistry.register('ALLChat')
class ALLChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        client = OpenAI(
            base_url=_resolve_base_url(self.model_name),
            api_key=_resolve_api_key(),
        )
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
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = AsyncOpenAI(
            base_url=_resolve_base_url(self.model_name),
            api_key=_resolve_api_key(),
        )
        chat_completion = await client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        max_tokens = max_tokens,
        temperature = temperature,
        )
        response = chat_completion.choices[0].message.content

        return response
    

@LLMRegistry.register('Deepseek')
class DSChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        client = OpenAI(base_url = os.environ.get("DS_URL"),
                        api_key = os.environ.get("DS_KEY"))
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
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = AsyncOpenAI(base_url = os.environ.get("DS_URL"),
                             api_key = os.environ.get("DS_KEY"),)
        chat_completion = await client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        max_tokens = max_tokens,
        temperature = temperature,
        )
        response = chat_completion.choices[0].message.content

        return response

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def achat(
    model: str,
    msg: List[Dict],):
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
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
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
    msg: List[Dict],):
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
    response = requests.post(request_url, headers=headers ,json=data)
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
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        return chat(self.model_name,messages)
    

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
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        client = OpenAI(base_url = os.environ.get("OPENROUTER_BASE_URL"),
                        api_key = os.environ.get("OPENROUTER_API_KEY"),)
        chat_completion = client.chat.completions.create(
        messages = messages,
        model = self.model_name,
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
