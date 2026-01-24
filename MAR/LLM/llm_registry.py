from typing import Optional
import threading
from class_registry import ClassRegistry

from MAR.LLM.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()
    _instance_lock = threading.Lock()
    _instances: dict[str, LLM] = {}

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o-mini"
        with cls._instance_lock:
            cached = cls._instances.get(model_name)
            if cached is not None:
                return cached

            if 'DeepSeek-V3' in model_name:
                model = cls.registry.get('Deepseek', model_name)
            else:
                model = cls.registry.get('ALLChat', model_name)

            cls._instances[model_name] = model
            return model
