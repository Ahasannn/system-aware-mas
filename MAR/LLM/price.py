from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.telemetry import LLMUsageTracker
import tiktoken
import threading
from functools import lru_cache
# GPT-4:  https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
# GPT3.5: https://platform.openai.com/docs/models/gpt-3-5
# DALL-E: https://openai.com/pricing
_GLOBAL_COUNTER_LOCK = threading.Lock()

@lru_cache(maxsize=32)
def _encoder_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def cal_token(model:str, text:str):
    encoder = _encoder_for_model(model or "gpt-4o")
    num_tokens = len(encoder.encode(text))
    return num_tokens

def truncate_text_for_model(text: str, max_tokens: int, model_name: str) -> str:
    if max_tokens <= 0:
        return ""
    encoder = _encoder_for_model(model_name or "gpt-4o")
    token_ids = encoder.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return encoder.decode(token_ids[-max_tokens:])

def cost_count(prompt, response, model_name):
    prompt_len: int
    completion_len: int
    price: float

    prompt_len = cal_token(model_name, prompt)
    completion_len = cal_token(model_name, response)
    price = 0
    if model_name in MODEL_PRICE.keys():
        prompt_price = MODEL_PRICE[model_name]["input"]
        completion_price = MODEL_PRICE[model_name]["output"]
        price = prompt_len * prompt_price / 1000000 + completion_len * completion_price / 1000000
    LLMUsageTracker.instance().record(cost=price, prompt_tokens=prompt_len, completion_tokens=completion_len)
    with _GLOBAL_COUNTER_LOCK:
        PromptTokens.instance().value += prompt_len
        CompletionTokens.instance().value += completion_len
        Cost.instance().value += price
    if model_name not in MODEL_PRICE.keys():
        return 0, prompt_len, completion_len

    # print(f"Prompt Tokens: {prompt_len}, Completion Tokens: {completion_len}")
    return price, prompt_len, completion_len

MODEL_PRICE = {
    "gpt-3.5-turbo-0125":{
        "input": 0.5,
        "output": 1.5
    },
    "gpt-3.5-turbo-1106":{
        "input": 1.0,
        "output": 2.0
    },
    "gpt-4-1106-preview":{
        "input": 10.0,
        "output": 30.0
    },
    "gpt-4o":{
        "input": 2.5,
        "output": 10.0
    },
    "gpt-4o-mini":{
        "input": 0.15,
        "output": 0.6
    },
    "claude-3-5-haiku-20241022":{
        "input": 0.8,
        "output": 4.0
    },
    "claude-3-5-sonnet-20241022":{
        "input": 3.0,
        "output": 15.0
    },
    "gemini-1.5-flash-latest":{
        "input": 0.15,
        "output": 0.60
    },
    "gemini-2.0-flash-thinking-exp":{
        "input": 4.0,
        "output": 16.0
    },
    "llama-3.3-70b-versatile":{
        "input": 0.2,
        "output": 0.2
    },
    "Meta-Llama-3.1-70B-Instruct":{
        "input": 0.2,
        "output": 0.2
    },
    "llama-3.1-70b-instruct":{
        "input": 0.2,
        "output": 0.2
    },
    'deepseek-chat':{
        'input': 0.27,
        'output': 1.1
    },
    'deepseek-ai/DeepSeek-V3':{
        'input': 0.27,
        'output': 1.1
    },
    # Test profile models (local/vLLM names)
    "meta-llama/Llama-3.2-3B-Instruct": {
        "input": 0.02,
        "output": 0.02,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "input": 0.20,
        "output": 0.20,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "input": 0.02,
        "output": 0.05,
    },
    # Model pool (vLLM local serving)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "input": 0.29,
        "output": 0.29,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "input": 0.03,
        "output": 0.11,
    },
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "input": 0.20,
        "output": 0.20,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "input": 0.02,
        "output": 0.05,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "input": 0.02,
        "output": 0.02,
    },
}
