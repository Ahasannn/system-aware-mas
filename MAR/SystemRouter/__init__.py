from importlib import import_module
from typing import Dict


__all__ = [
    "SystemRouterEnv",
    "SystemAwareRouter",
    "PromptStrategy",
    "SystemRouterTrainer",
    "LengthEstimator",
    "LengthEstimatorBundle",
    "LatencyEstimator",
    "LatencyEstimatorBundle",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "SystemRouterEnv": ".env",
    "SystemAwareRouter": ".system_aware_router",
    "PromptStrategy": ".prompt_strategies",
    "SystemRouterTrainer": ".trainer",
    "LengthEstimator": ".length_estimator",
    "LengthEstimatorBundle": ".length_estimator",
    "LatencyEstimator": ".latency_estimator",
    "LatencyEstimatorBundle": ".latency_estimator",
}


def __getattr__(name: str):
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
