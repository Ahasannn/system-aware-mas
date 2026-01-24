import json
import runpy
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from MAR.SystemRouter.metrics_watcher import model_metrics
from MAR.Utils.offline_embeddings import load_query_embeddings, load_role_embeddings

class SemanticEncoder(nn.Module):
    """
    384-d semantic encoder backed by `sentence-transformers/all-MiniLM-L6-v2`.
    """

    def __init__(self, device: torch.device, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(model_name, device=str(device))
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

    def forward(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = self.model.encode(sentences, convert_to_tensor=True, device=str(self.device))
        return embeddings.to(self.device).clone()


class SystemAwareRouter(nn.Module):
    """
    Hierarchical CMDP router:
      - Planner: selects topology + role set at t=0.
      - Executor: selects (LLM, strategy) per role during runtime.

    Planner state: query embedding only.
    Executor state: [query_emb || role_emb || B_rem || system_metrics].
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        topologies: Optional[List[str]] = None,
        role_sets: Optional[List[List[str]]] = None,
        role_domain: str = "Code",
        embedding_dim: int = 384,
        hidden_dim: int = 128,
        fixed_budget_sec: float = 60.0,
        lambda_init: float = 0.5,
        device: Optional[torch.device] = None,
        query_embeddings_csv: Optional[str] = None,
        role_embeddings_csv: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Defaults align with the local vLLM test pool (see `config_test.json` / `MAR/LLM/llm_profile_full.json`).
        self.models = models or [
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
        self.strategies = strategies or ["Flash", "Concise", "DeepThink"]
        self.topologies = topologies or [item["Name"] for item in _load_reasoning_profile()]
        self.role_domain = role_domain
        self.role_profiles = _load_role_profiles(role_domain)
        role_names = list(self.role_profiles.keys())
        self.role_sets = role_sets or _default_role_sets(role_names)
        self.role_set_names = ["-".join(role_set) for role_set in self.role_sets]
        self.fixed_budget_sec = float(fixed_budget_sec)

        self.system_metric_keys = [
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
            "ttft_avg",
            "itl_avg",
            "e2e_avg",
        ]
        self.latency_dim = len(self.models) * len(self.system_metric_keys)  # system metrics per model
        self.encoder = SemanticEncoder(device=self.device)
        embeddings_dir = Path(__file__).resolve().parents[2] / "Datasets" / "embeddings"
        query_csv = query_embeddings_csv or str(embeddings_dir / "query_embeddings.csv")
        roles_csv = role_embeddings_csv or str(embeddings_dir / "role_embeddings.csv")
        self.offline_query_embeddings = load_query_embeddings(
            query_csv, device=self.device, dtype=torch.float32
        )
        self.offline_role_embeddings = load_role_embeddings(roles_csv, device=self.device, dtype=torch.float32)

        if embedding_dim != self.encoder.embedding_dim:
            raise ValueError(
                f"embedding_dim={embedding_dim} does not match encoder output "
                f"{self.encoder.embedding_dim}. Use embedding_dim={self.encoder.embedding_dim}."
            )
        self.embedding_dim = self.encoder.embedding_dim

        # Planner policy over topology + role set
        self.planner_backbone = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.topology_head = nn.Linear(hidden_dim, len(self.topologies))
        self.role_head = nn.Linear(hidden_dim, len(self.role_sets))

        # Executor policy over (LLM, strategy)
        executor_state_dim = self.embedding_dim + self.embedding_dim + 1 + self.latency_dim
        self.executor_backbone = nn.Sequential(
            nn.Linear(executor_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.model_head = nn.Linear(hidden_dim, len(self.models))
        self.strategy_head = nn.Linear(hidden_dim, len(self.strategies))
        self.value_head = nn.Linear(hidden_dim, 1)

        self.lagrange_multiplier = nn.Parameter(torch.tensor(lambda_init, device=self.device))
        self.to(self.device)

    def encode_query(
        self,
        query: str,
        query_id: Optional[object] = None,
        dataset_name: Optional[str] = None,
    ) -> torch.Tensor:
        dataset_key = str(dataset_name).strip().lower() if dataset_name is not None else ""
        if query_id is not None and dataset_key:
            try:
                query_id_int = int(query_id)
            except (TypeError, ValueError):
                query_id_int = None
            if query_id_int is not None:
                cached = self.offline_query_embeddings.get((dataset_key, query_id_int))
                if cached is not None:
                    return cached
        return self.encoder([query])[0]

    def encode_role(self, role: str) -> torch.Tensor:
        cached = self.offline_role_embeddings.get(role)
        if cached is not None:
            return cached
        profile = self.role_profiles.get(role)
        if profile is None:
            return self.encoder([role])[0]
        return self.encoder([json.dumps(profile)])[0]

    def plan_graph(
        self,
        query: str,
        deterministic: bool = False,
        query_id: Optional[object] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Union[str, int, torch.Tensor, List[str]]]:
        query_embedding = self.encode_query(query, query_id=query_id, dataset_name=dataset_name)
        hidden = self.planner_backbone(query_embedding.unsqueeze(0))
        topology_logits = self.topology_head(hidden)
        role_logits = self.role_head(hidden)

        topology_probs = topology_logits.softmax(dim=-1)
        role_probs = role_logits.softmax(dim=-1)

        if deterministic:
            topology_idx = topology_probs.argmax(dim=-1)
            role_idx = role_probs.argmax(dim=-1)
        else:
            topology_idx = torch.distributions.Categorical(topology_probs).sample()
            role_idx = torch.distributions.Categorical(role_probs).sample()

        topology_id = int(topology_idx.item())
        role_set_id = int(role_idx.item())
        return {
            "query_embedding": query_embedding,
            "topology_index": topology_idx,
            "role_index": role_idx,
            "topology_probs": topology_probs,
            "role_probs": role_probs,
            "topology_name": self.topologies[topology_id],
            "role_set_name": self.role_set_names[role_set_id],
            "role_set": self.role_sets[role_set_id],
        }

    def planner_forward(self, query_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.planner_backbone(query_embeddings)
        topology_probs = self.topology_head(hidden).softmax(dim=-1)
        role_probs = self.role_head(hidden).softmax(dim=-1)
        return {"topology_probs": topology_probs, "role_probs": role_probs}

    def assemble_executor_state(
        self,
        query_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        budget_remaining: float,
        system_state_vector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        budget_tensor = torch.tensor([budget_remaining], device=self.device, dtype=query_embedding.dtype)
        state_tensor = torch.cat([query_embedding, role_embedding, budget_tensor, system_state_vector], dim=0)
        return {
            "state": state_tensor,
            "budget_remaining": budget_tensor,
        }

    def get_executor_action(
        self, executor_state: Union[torch.Tensor, Dict[str, torch.Tensor]], deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        state_tensor = self._as_state_tensor(executor_state)
        hidden = self.executor_backbone(state_tensor)
        model_logits = self.model_head(hidden)
        strategy_logits = self.strategy_head(hidden)
        model_probs = model_logits.softmax(dim=-1)
        strategy_probs = strategy_logits.softmax(dim=-1)

        if deterministic:
            model_idx = model_probs.argmax(dim=-1)
            strategy_idx = strategy_probs.argmax(dim=-1)
        else:
            model_idx = torch.distributions.Categorical(model_probs).sample()
            strategy_idx = torch.distributions.Categorical(strategy_probs).sample()

        value = self.value_head(hidden).squeeze(-1)
        return {
            "model_index": model_idx,
            "strategy_index": strategy_idx,
            "model_probs": model_probs,
            "strategy_probs": strategy_probs,
            "value": value,
        }

    def estimate_initial_budget(self, query: str) -> float:
        """
        Fixed latency budget placeholder (set to a high value for now).
        """
        return self.fixed_budget_sec

    def get_system_metrics(self) -> Dict[str, Dict[str, float]]:
        return {model: dict(model_metrics.get(model, {})) for model in self.models}

    def flatten_system_metrics(self, metrics: Dict[str, Dict[str, float]], dtype: torch.dtype) -> torch.Tensor:
        values: List[float] = []
        for model in self.models:
            snap = metrics.get(model, {})
            for key in self.system_metric_keys:
                values.append(float(snap.get(key, 0.0)))
        return torch.tensor(values, device=self.device, dtype=dtype)

    def get_system_state_vector(self, dtype: torch.dtype) -> torch.Tensor:
        return self.flatten_system_metrics(self.get_system_metrics(), dtype=dtype)

    def compute_executor_reward(
        self,
        semantic_quality: Union[float, torch.Tensor],
        latency: Union[float, torch.Tensor],
        budget_remaining: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        quality = self._to_tensor(semantic_quality)
        latency_tensor = self._to_tensor(latency).to(self.device)
        budget_tensor = self._to_tensor(budget_remaining).clamp_min(1e-3)

        penalty = self.latency_penalty(latency_tensor, budget_tensor)
        lambda_value = F.softplus(self.lagrange_multiplier)
        return quality - lambda_value * penalty

    def latency_penalty(self, latency: torch.Tensor, budget_remaining: torch.Tensor) -> torch.Tensor:
        over_budget = torch.relu(latency - budget_remaining)
        return over_budget / budget_remaining

    def _as_state_tensor(self, state: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(state, dict):
            state_tensor = state["state"]
        else:
            state_tensor = state
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor.to(self.device)

    def _to_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else torch.tensor([value], device=self.device)


def _load_role_profiles(domain: str) -> Dict[str, Dict[str, object]]:
    roles_dir = Path(__file__).resolve().parents[2] / "MAR" / "Roles" / domain
    if not roles_dir.is_dir():
        raise RuntimeError(f"Role domain not found: {roles_dir}")

    profiles: Dict[str, Dict[str, object]] = {}
    for path in sorted(roles_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        name = profile.get("Name") or path.stem
        profiles[str(name)] = profile

    if not profiles:
        raise RuntimeError(f"No role profiles found under {roles_dir}")
    return profiles


def _default_role_sets(role_names: List[str]) -> List[List[str]]:
    base_sets = [
        ["ProjectManager", "ProgrammingExpert", "TestAnalyst"],
        ["PlanSolver", "ProgrammingExpert"],
        ["ProgrammingExpert", "TestAnalyst"],
        ["AlgorithmDesigner", "ProgrammingExpert"],
        ["BugFixer", "ProgrammingExpert"],
        ["ReflectProgrammer", "ProgrammingExpert"],
    ]
    for role in role_names:
        base_sets.append([role])

    seen = set()
    filtered: List[List[str]] = []
    for role_set in base_sets:
        if all(role in role_names for role in role_set):
            key = tuple(role_set)
            if key not in seen:
                seen.add(key)
                filtered.append(role_set)
    if not filtered:
        raise RuntimeError("No valid role sets could be constructed from role profiles.")
    return filtered


def _load_reasoning_profile() -> List[Dict[str, object]]:
    profile_path = Path(__file__).resolve().parents[2] / "MAR" / "Agent" / "reasoning_profile.py"
    if not profile_path.is_file():
        raise RuntimeError(f"Reasoning profile not found: {profile_path}")
    data = runpy.run_path(str(profile_path))
    profile = data.get("reasoning_profile")
    if not isinstance(profile, list):
        raise RuntimeError("Reasoning profile file did not define a list named reasoning_profile.")
    return profile
