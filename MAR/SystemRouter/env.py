import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from MAR.SystemRouter.metrics_watcher import start_metrics_watcher
from MAR.SystemRouter.system_aware_router import SystemAwareRouter
from MAR.SystemRouter.system_router_graph import SystemRouterGraph
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import get_kwargs


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _get_test_config() -> Dict[str, object]:
    candidates = [
        _project_root() / "config_test.json",
        _project_root() / "MAR" / "LLM" / "llm_profile_full.json",
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


def _extract_model_base_urls(data: object) -> Dict[str, str]:
    if not isinstance(data, dict):
        return {}

    if "model_base_urls" in data and isinstance(data.get("model_base_urls"), dict):
        data = data["model_base_urls"]

    normalized: Dict[str, str] = {}
    if not isinstance(data, dict):
        return {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


@lru_cache(maxsize=1)
def _get_model_base_urls() -> Dict[str, str]:
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


def _metrics_url_from_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
        base = base.rstrip("/")
    return f"{base}/metrics"


def _build_metrics_url_map(models: List[str]) -> Dict[str, str]:
    urls: Dict[str, str] = {}
    for model in models:
        base_url = _resolve_base_url(model)
        if not base_url:
            logger.warning("[Metrics] No base URL for model: {}", model)
            continue
        metrics_url = _metrics_url_from_base(base_url)
        urls[model] = metrics_url
        logger.debug("[Metrics] {} -> {}", model, metrics_url)
    if not urls:
        logger.warning("[Metrics] No metrics URLs built! Models: {}", models)
    return urls


class SystemRouterEnv:
    """
    Hierarchical environment wrapper for planner + executor routing.
    """

    def __init__(
        self,
        router: SystemAwareRouter,
        max_tokens: int = 256,
        prompt_file: Optional[str] = None,
        metrics_interval: float = 1.0,
        metrics_url_map: Optional[Dict[str, str]] = None,
        request_timeout: float = 600.0,
        quality_fn: Optional[Callable[[str, Optional[List[str]], Optional[Any]], Tuple[Union[float, torch.Tensor], Dict[str, object]]]] = None,
    ) -> None:
        self.router = router
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.router_lock = threading.Lock()
        self.quality_fn = quality_fn
        self.prompt_file = prompt_file or str(
            Path(__file__).resolve().parents[2] / "MAR" / "Roles" / "FinalNode" / "mbpp.json"
        )
        self.metrics_url_map = metrics_url_map or _build_metrics_url_map(router.models)
        self.metrics_thread = None
        if self.metrics_url_map:
            logger.info("[Env] Starting metrics watcher with {} models", len(self.metrics_url_map))
            self.metrics_thread = start_metrics_watcher(self.metrics_url_map, interval=metrics_interval)
        else:
            logger.warning("[Env] No metrics URL map - metrics watcher NOT started. Router models: {}", router.models)

    def step(
        self,
        query: str,
        tests: Optional[List[str]] = None,
        deterministic: bool = False,
        latency_seed: Optional[str] = None,
        query_id: Optional[object] = None,
        dataset_name: Optional[str] = None,
        sample: Optional[Any] = None,
        quality_fn: Optional[
            Callable[[str, Optional[List[str]], Optional[Any]], Tuple[Union[float, torch.Tensor], Dict[str, object]]]
        ] = None,
    ) -> Dict[str, Union[str, float, torch.Tensor, Dict[str, float], List[dict]]]:
        """
        Run one hierarchical episode: planner picks topology/roles, executor runs each role.
        Tests are required unless a quality_fn is supplied.
        """
        scorer = quality_fn or self.quality_fn
        if not scorer and not tests:
            raise ValueError("Tests are required for quality scoring; no fallback is used.")

        with self.router_lock:
            plan = self.router.plan_graph(
                query,
                deterministic=deterministic,
                query_id=query_id,
                dataset_name=dataset_name,
            )

        budget_total = self.router.estimate_initial_budget(query)

        role_set = plan["role_set"]
        graph_kwargs = get_kwargs(plan["topology_name"], len(role_set))
        node_kwargs = graph_kwargs.get("node_kwargs")
        if not node_kwargs or len(node_kwargs) != len(role_set):
            node_kwargs = [{} for _ in role_set]
        for kwargs in node_kwargs:
            kwargs.setdefault("max_tokens", self.max_tokens)
            kwargs.setdefault("request_timeout", self.request_timeout)
        graph_kwargs["node_kwargs"] = node_kwargs
        num_rounds = graph_kwargs.pop("num_rounds", 1)
        llm_names = [self.router.models[0] for _ in role_set]
        graph = SystemRouterGraph(
            domain=self.router.role_domain,
            llm_names=llm_names,
            agent_names=role_set,
            decision_method="FinalRefer",
            prompt_file=self.prompt_file,
            reasoning_name=plan["topology_name"],
            runtime_llm_assignment=False,
            **graph_kwargs,
        )
        graph.max_tokens = self.max_tokens
        graph_result = graph.run_with_policy(
            inputs={"query": query},
            router=self.router,
            query_embedding=plan["query_embedding"],
            budget_total=budget_total,
            num_rounds=num_rounds,
            deterministic=deterministic,
            router_lock=self.router_lock,
        )

        final_response = graph_result["final_response"]
        executor_transitions = graph_result["executor_transitions"]
        token_tally = graph_result["token_counts"]
        workflow_latency = graph_result["workflow_latency_seconds"]
        llm_elapsed_seconds = float(graph_result.get("llm_elapsed_seconds", workflow_latency))

        if scorer:
            quality_value, quality_meta = scorer(final_response, tests, sample)
            quality = self._normalize_quality(quality_value)
        else:
            quality, quality_meta = self._score_quality(final_response, tests)
        for transition in executor_transitions:
            transition["quality"] = float(quality.detach().cpu().item())
            transition["workflow_latency_seconds"] = workflow_latency

        planner_transition = {
            "state": plan["query_embedding"],
            "action": {
                "topology_index": plan["topology_index"],
                "role_index": plan["role_index"],
            },
            "quality": float(quality.detach().cpu().item()),
        }

        return {
            "response": final_response,
            "workflow_latency_seconds": workflow_latency,
            "llm_elapsed_seconds": llm_elapsed_seconds,
            "budget_total": budget_total,
            "token_counts": token_tally,
            "quality": float(quality.detach().cpu().item()),
            "quality_is_solved": quality_meta.get("is_solved"),
            "quality_feedback": quality_meta.get("feedback"),
            "quality_pred": quality_meta.get("pred"),
            "quality_gold": quality_meta.get("gold"),
            "topology": plan["topology_name"],
            "role_set": plan["role_set_name"],
            "code_response": final_response,
            "planner_transition": planner_transition,
            "executor_transitions": executor_transitions,
        }

    def _score_quality(self, response: str, tests: List[str]) -> Tuple[torch.Tensor, Dict[str, Union[bool, str]]]:
        """
        Code execution (MBPP-style). Requires tests.
        """
        code = self._extract_code(response)
        is_solved, feedback, _ = PyExecutor().execute(code, list(tests), timeout=30, verbose=False)
        quality = torch.tensor([1.0 if is_solved else 0.0], device=self.router.device)
        return quality, {"is_solved": bool(is_solved), "feedback": feedback}

    def _normalize_quality(self, quality: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(quality, torch.Tensor):
            if quality.dim() == 0:
                quality = quality.unsqueeze(0)
            return quality.to(self.router.device)
        return torch.tensor([float(quality)], device=self.router.device)

    def _extract_code(self, response: str) -> str:
        pattern = r"```python(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()
