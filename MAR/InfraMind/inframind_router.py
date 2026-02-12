"""
InfraMind Router — Budget-Aware Hierarchical CMDP Router.

Architecture:
  - **Planner**: MAS-based VAE+GFusion modules (TaskClassifier, CollabDeterminer,
    NumDeterminer, RoleAllocation) conditioned on latency budget via
    Budget-Conditioned Feature Modulation (BCFM / FiLM).
    Selects topology, agent count, and roles.  NO upfront LLM selection.

  - **Executor**: Per-node MLP that selects (model, strategy) at runtime based on
    query embedding, role embedding, remaining budget, and live system metrics.
"""

import json
import math
import os
import runpy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer

from MAR.InfraMind.metrics_watcher import model_metrics
from MAR.Utils.offline_embeddings import load_query_embeddings, load_role_embeddings


# ---------------------------------------------------------------------------
# Shared building blocks (same as MAS baseline)
# ---------------------------------------------------------------------------

_STD2 = 0.1
_VAR2 = _STD2 * _STD2
_LOG_VAR2 = math.log(_VAR2)


class GFusion(nn.Module):
    """Graph Fusion Module — cross-attention between two embedding sets."""

    def __init__(self, d_model: int = 384):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(x)
        K = self.key_proj(y)
        V = self.value_proj(y)
        attn = torch.matmul(Q, K.transpose(0, 1)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        ctx = F.normalize(torch.matmul(attn, V), p=2, dim=1)
        return self.out_proj(x + ctx)


class VAE(nn.Module):
    """Variational autoencoder for latent embedding space."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var) * _STD2
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc4(F.relu(self.fc3(z)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), z, mu, log_var


def vae_loss_function(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 - _LOG_VAR2 + log_var - (mu.pow(2) + log_var.exp()) / _VAR2)
    return mse + kld


# ---------------------------------------------------------------------------
# MAS planner modules (imported verbatim from baseline)
# ---------------------------------------------------------------------------

class TaskClassifier(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, temp: float = 1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.query_encoder = nn.Linear(input_dim, hidden_dim)
        self.task_encoder = nn.Linear(input_dim, hidden_dim)
        self.temp = temp

    def forward(self, queries: torch.Tensor, tasks: torch.Tensor):
        q = F.normalize(self.query_encoder(queries), p=2, dim=1)
        t = F.normalize(self.task_encoder(tasks), p=2, dim=1)
        scores = F.softmax(torch.matmul(q, t.T) / self.temp, dim=1)
        return torch.argmax(scores, dim=1), scores, q


class CollabDeterminer(nn.Module):
    def __init__(self, input_dim: int = 384, context_input_dim: int = 384, hidden_dim: int = 64, temp: float = 1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.collab_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        self.context_encoder = VAE(context_input_dim, hidden_dim, hidden_dim)
        self.collab_context_encoder = GFusion(d_model=hidden_dim)
        self.temp = temp

    def forward(self, collabs: torch.Tensor, contexts: torch.Tensor):
        collab_hat, collab_z, collab_mu, collab_logvar = self.collab_encoder(collabs)
        collab_z = F.normalize(collab_z, p=2, dim=1)
        context_hat, context_z, context_mu, context_logvar = self.context_encoder(contexts)
        context_z = F.normalize(context_z, p=2, dim=1)

        scores = torch.softmax(torch.matmul(context_z, collab_z.T) / self.temp, dim=1)
        vae_loss = (vae_loss_function(collab_hat, collabs, collab_mu, collab_logvar)
                    + vae_loss_function(context_hat, contexts, context_mu, context_logvar))

        scores_cumsum = torch.cumsum(scores, dim=1)
        random_num = torch.rand([scores.size(0), 1], device=self.device)
        selected_index = (scores_cumsum > random_num).float().argmax(dim=1)
        log_probs = torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1)
        collab_embedding = collab_z[selected_index]
        return selected_index, log_probs, collab_embedding, vae_loss

    def evaluate(self, collabs: torch.Tensor, contexts: torch.Tensor, chosen_idx: int):
        """Re-run forward pass and evaluate log_prob of a given action index."""
        collab_hat, collab_z, collab_mu, collab_logvar = self.collab_encoder(collabs)
        collab_z = F.normalize(collab_z, p=2, dim=1)
        context_hat, context_z, context_mu, context_logvar = self.context_encoder(contexts)
        context_z = F.normalize(context_z, p=2, dim=1)

        scores = torch.softmax(torch.matmul(context_z, collab_z.T) / self.temp, dim=1)
        vae_loss = (vae_loss_function(collab_hat, collabs, collab_mu, collab_logvar)
                    + vae_loss_function(context_hat, contexts, context_mu, context_logvar))

        log_prob = torch.log(scores[0, chosen_idx]).unsqueeze(0).unsqueeze(0)
        collab_embedding = collab_z[chosen_idx].unsqueeze(0)
        return log_prob, collab_embedding, vae_loss


class NumDeterminer(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, max_agent: int = 6, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE(input_dim, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.max_agent = max_agent

    def forward(self, queries: torch.Tensor):
        x_hat, z, mu, log_var = self.vae(queries)
        z = F.normalize(z, p=2, dim=1)
        difficulty = torch.sigmoid(self.fc(z))
        agent_num_float = difficulty * self.max_agent
        agent_num_int = torch.clamp(torch.round(agent_num_float), 1, self.max_agent).int()
        vae_loss = vae_loss_function(x_hat, queries, mu, log_var)
        return agent_num_int, agent_num_float, vae_loss


class RoleAllocation(nn.Module):
    def __init__(self, input_dim: int = 384, context_input_dim: int = 128, hidden_dim: int = 64, temp: float = 1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_role_embedding = torch.zeros([1, hidden_dim], device=self.device, requires_grad=True)
        self.role_encoder = VAE(input_dim, hidden_dim, hidden_dim)
        self.context_encoder = nn.Linear(context_input_dim + hidden_dim, hidden_dim)
        self.role_context_encoder = GFusion(d_model=hidden_dim)
        self.temp = temp

    def forward(self, roles_list: List[torch.Tensor], contexts: torch.Tensor, agent_num_int: torch.Tensor):
        selected_roles_idx: List[List[torch.Tensor]] = []
        log_probs = torch.zeros([contexts.size(0), 1], device=self.device)
        summary_role_list = []

        for i, roles in enumerate(roles_list):
            selected_roles_idx.append([])
            role_hat, role_z, role_mu, role_log_var = self.role_encoder(roles)
            role_embedding = F.normalize(role_z, p=2, dim=1)
            cur_vae_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var)
            vae_loss = cur_vae_loss if i == 0 else vae_loss + cur_vae_loss

            current_role_embedding = self.init_role_embedding
            history_role_embedding = self.init_role_embedding

            for j in range(agent_num_int[i]):
                history_role_embedding = history_role_embedding + current_role_embedding
                history_role_embedding = F.layer_norm(history_role_embedding, history_role_embedding.shape[1:])
                ctx = self.context_encoder(torch.cat([contexts[i].unsqueeze(0), history_role_embedding], dim=1))
                ctx = F.normalize(ctx, p=2, dim=1)
                scores = torch.softmax(torch.matmul(ctx, role_embedding.T) / self.temp, dim=1)
                scores_cumsum = torch.cumsum(scores, dim=1)
                random_num = torch.rand([scores.size(0), 1], device=self.device)
                selected_index = (scores_cumsum > random_num).float().argmax(dim=1)
                log_probs[i][0] = log_probs[i][0] + torch.log(scores[torch.arange(scores.size(0)), selected_index]).unsqueeze(1)
                current_role_embedding = role_embedding[selected_index]
                selected_roles_idx[-1].append(selected_index)
            summary_role_list.append(history_role_embedding)

        summary_role = torch.cat(summary_role_list, dim=0)
        return selected_roles_idx, log_probs, summary_role, vae_loss / len(roles_list)

    def evaluate(
        self,
        roles_list: List[torch.Tensor],
        contexts: torch.Tensor,
        agent_num_int: torch.Tensor,
        given_role_indices: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-run forward pass and evaluate log_prob of given role indices."""
        log_probs = torch.zeros([contexts.size(0), 1], device=self.device)

        for i, roles in enumerate(roles_list):
            role_hat, role_z, role_mu, role_log_var = self.role_encoder(roles)
            role_embedding = F.normalize(role_z, p=2, dim=1)
            cur_vae_loss = vae_loss_function(role_hat, roles, role_mu, role_log_var)
            vae_loss = cur_vae_loss if i == 0 else vae_loss + cur_vae_loss

            current_role_embedding = self.init_role_embedding
            history_role_embedding = self.init_role_embedding

            for j in range(agent_num_int[i]):
                history_role_embedding = history_role_embedding + current_role_embedding
                history_role_embedding = F.layer_norm(history_role_embedding, history_role_embedding.shape[1:])
                ctx = self.context_encoder(torch.cat([contexts[i].unsqueeze(0), history_role_embedding], dim=1))
                ctx = F.normalize(ctx, p=2, dim=1)
                scores = torch.softmax(torch.matmul(ctx, role_embedding.T) / self.temp, dim=1)
                selected_index = given_role_indices[i][j]
                log_probs[i][0] = log_probs[i][0] + torch.log(scores[0, selected_index])
                current_role_embedding = role_embedding[selected_index].unsqueeze(0)

        return log_probs, vae_loss / len(roles_list)


# ---------------------------------------------------------------------------
# Budget-Conditioned Feature Modulation (BCFM)
# ---------------------------------------------------------------------------

class BudgetConditioner(nn.Module):
    """
    Budget-Conditioned Feature Modulation (BCFM).

    Applies Feature-wise Linear Modulation (FiLM) to condition the semantic
    query embedding on the available latency budget.  The scale parameter γ
    is initialized near identity (centered at 1) so the module degrades
    gracefully to standard MAS behaviour when budget information is
    uninformative.

    Args:
        embed_dim: Dimensionality of the query embedding (default 384).
        hidden_dim: Width of the budget encoder MLP (default 64).
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.budget_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = nn.Linear(hidden_dim, embed_dim)  # γ
        self.shift = nn.Linear(hidden_dim, embed_dim)  # β

    def forward(self, query_emb: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: ``(B, D)`` semantic query embedding.
            budget:    ``(B, 1)`` budget in seconds (raw, not normalized).
        Returns:
            ``(B, D)`` budget-conditioned query embedding.
        """
        h = self.budget_encoder(budget)
        gamma = 1.0 + self.scale(h)   # centered at identity
        beta = self.shift(h)
        return gamma * query_emb + beta


# ---------------------------------------------------------------------------
# Semantic Encoder
# ---------------------------------------------------------------------------

class SemanticEncoder(nn.Module):
    """384-d encoder backed by ``sentence-transformers/all-MiniLM-L6-v2``."""

    def __init__(self, device: torch.device, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(model_name, device=str(device))
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

    def forward(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.model.encode(sentences, convert_to_tensor=True, device=str(self.device)).to(self.device).clone()


# ---------------------------------------------------------------------------
# InfraMind Router
# ---------------------------------------------------------------------------

# Task-domain → index in tasks_profile
_DOMAIN_TASK_INDEX = {"Math": 0, "Commonsense": 1, "Code": 2}


class InfraMindRouter(nn.Module):
    """
    Hierarchical CMDP router.

    **Planner** (MAS architecture + BCFM):
        Budget-conditioned query embedding → TaskClassifier → CollabDeterminer
        → NumDeterminer → RoleAllocation.  Returns topology, agent count, and
        roles together with differentiable log-probabilities for REINFORCE.
        No LLM selection at planning time.

    **Executor** (infrastructure-aware MLP):
        Per-node runtime selection of (model, strategy) based on query embedding,
        role embedding, remaining budget, and live vLLM system metrics.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        role_domain: str = "Code",
        embedding_dim: int = 384,
        planner_hidden_dim: int = 64,
        executor_hidden_dim: int = 128,
        max_agent: int = 6,
        lambda_init: float = 0.5,
        device: Optional[torch.device] = None,
        query_embeddings_csv: Optional[str] = None,
        role_embeddings_csv: Optional[str] = None,
        latency_predictor_path: Optional[str] = None,
        length_predictor_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model pool for executor
        self.models = models or _load_models_from_profile() or [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ]
        self.strategies = strategies or ["Flash", "Concise", "DeepThink"]
        self.role_domain = role_domain
        self.max_agent = max_agent
        self.embedding_dim = embedding_dim

        # ---- Semantic encoder & offline caches ----------------------------
        self.encoder = SemanticEncoder(device=self.device)
        embeddings_dir = Path(__file__).resolve().parents[2] / "Datasets" / "embeddings"
        self.offline_query_embeddings = load_query_embeddings(
            query_embeddings_csv or str(embeddings_dir / "query_embeddings.csv"),
            device=self.device, dtype=torch.float32,
        )
        self.offline_role_embeddings = load_role_embeddings(
            role_embeddings_csv or str(embeddings_dir / "role_embeddings.csv"),
            device=self.device, dtype=torch.float32,
        )

        # ---- Role profiles (for executor role embedding lookup) -----------
        self.role_profiles = _load_role_profiles(role_domain)

        # ---- Tasks, collabs, roles metadata (for planner) -----------------
        self._tasks_profile = _load_tasks_profile()
        self._reasoning_profile = _load_reasoning_profile()
        self._task_role_database: Optional[Dict] = None
        self._task_role_emb: Optional[Dict] = None
        self._cached_tasks_emb: Optional[torch.Tensor] = None
        self._cached_collabs_emb: Optional[torch.Tensor] = None

        # ---- Budget-Conditioned Feature Modulation (BCFM) -----------------
        self.budget_conditioner = BudgetConditioner(
            embed_dim=embedding_dim,
            hidden_dim=planner_hidden_dim,
        )

        # ---- MAS Planner modules (no LLMRouter) --------------------------
        self.task_classifier = TaskClassifier(
            input_dim=embedding_dim, hidden_dim=planner_hidden_dim,
            temp=0.5, device=self.device,
        )
        self.collab_determiner = CollabDeterminer(
            input_dim=embedding_dim, context_input_dim=embedding_dim,
            hidden_dim=planner_hidden_dim, temp=0.8, device=self.device,
        )
        self.num_determiner = NumDeterminer(
            input_dim=embedding_dim, hidden_dim=planner_hidden_dim,
            max_agent=max_agent, device=self.device,
        )
        self.role_allocation = RoleAllocation(
            input_dim=embedding_dim,
            context_input_dim=2 * planner_hidden_dim,
            hidden_dim=planner_hidden_dim, temp=0.5, device=self.device,
        )

        # ---- Latency / length predictors (for executor) -------------------
        self.latency_predictor: Optional[object] = None
        self.length_predictor: Optional[object] = None
        if latency_predictor_path and length_predictor_path:
            self.latency_predictor, self.length_predictor = _load_predictors(
                latency_predictor_path, length_predictor_path, self.encoder, self.device,
            )
            logger.info("[Router] Predictors loaded: latency={}, length={}", latency_predictor_path, length_predictor_path)
        else:
            logger.warning("[Router] No predictors loaded — system state will be zeros")

        # ---- Executor policy (model + strategy selection) -----------------
        self.latency_dim = len(self.models) * (len(self.strategies) + 2)
        executor_state_dim = embedding_dim + embedding_dim + 1 + self.latency_dim
        self.executor_backbone = nn.Sequential(
            nn.Linear(executor_state_dim, executor_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(executor_hidden_dim),
            nn.Linear(executor_hidden_dim, executor_hidden_dim),
            nn.ReLU(),
        )
        self.model_head = nn.Linear(executor_hidden_dim, len(self.models))
        self.strategy_head = nn.Linear(executor_hidden_dim, len(self.strategies))
        self.value_head = nn.Linear(executor_hidden_dim, 1)

        # Lagrange multiplier for CMDP latency constraint
        self.lagrange_multiplier = nn.Parameter(torch.tensor(lambda_init, device=self.device))
        self.to(self.device)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def encode_query(self, query: str, query_id: Optional[object] = None, dataset_name: Optional[str] = None) -> torch.Tensor:
        dataset_key = str(dataset_name).strip().lower() if dataset_name else ""
        if query_id is not None and dataset_key:
            try:
                qid = int(query_id)
                cached = self.offline_query_embeddings.get((dataset_key, qid))
                if cached is not None:
                    return cached
            except (TypeError, ValueError):
                pass
        return self.encoder([query])[0]

    def encode_role(self, role: str) -> torch.Tensor:
        cached = self.offline_role_embeddings.get(role)
        if cached is not None:
            return cached
        profile = self.role_profiles.get(role)
        if profile is None:
            return self.encoder([role])[0]
        return self.encoder([json.dumps(profile)])[0]

    # ------------------------------------------------------------------
    # Planner: budget-conditioned MAS pipeline
    # ------------------------------------------------------------------

    def _ensure_planner_caches(self) -> None:
        """Lazily compute and cache static embeddings used by the planner."""
        if self._task_role_database is None:
            self._task_role_database, self._task_role_emb = self._encode_roles()
        if self._cached_tasks_emb is None:
            task_texts = [f"{t['Name']} : {t['Description']}" for t in self._tasks_profile]
            self._cached_tasks_emb = self.encoder(task_texts).to(self.device)
        if self._cached_collabs_emb is None:
            collab_texts = [f"{c['Name']} : {c['Description']}" for c in self._reasoning_profile]
            self._cached_collabs_emb = self.encoder(collab_texts).to(self.device)

    def _encode_roles(self) -> Tuple[Dict, Dict]:
        """Load role JSON files and compute text embeddings.

        IMPORTANT: Only loads roles from self.role_domain to prevent
        cross-domain contamination during role selection.
        """
        task_role_database: Dict[str, List[Dict]] = {}
        task_role_emb: Dict[str, torch.Tensor] = {}
        roles_root = Path(__file__).resolve().parents[1] / "Roles"

        # Only load roles from the specified domain
        task_dir = roles_root / self.role_domain
        if not task_dir.is_dir():
            raise RuntimeError(f"Role domain directory not found: {task_dir}")

        task_role_database[self.role_domain] = []
        texts: List[str] = []
        for role_file in sorted(task_dir.glob("*.json")):
            with role_file.open("r", encoding="utf-8") as f:
                profile = json.load(f)
            task_role_database[self.role_domain].append(profile)
            texts.append(json.dumps(profile))

        if texts:
            task_role_emb[self.role_domain] = self.encoder(texts).to(self.device)
        else:
            raise RuntimeError(f"No role profiles found in {task_dir}")

        logger.info(
            "[Router] Loaded {} roles from {} domain only",
            len(task_role_database[self.role_domain]),
            self.role_domain,
        )
        return task_role_database, task_role_emb

    def plan_graph(
        self,
        query: str,
        budget_total: float = 60.0,
        deterministic: bool = False,
        query_id: Optional[object] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Run the full planner pipeline:
            query → BCFM(budget) → TaskClassifier → CollabDeterminer
            → NumDeterminer → RoleAllocation.

        Returns a dict with:
            query_embedding, topology_name, role_names, agent_count,
            planner_log_probs, planner_vae_loss, task_probs, task_name,
            topology_index, agent_num_float.
        """
        self._ensure_planner_caches()

        # 1. Encode query
        query_embedding = self.encode_query(query, query_id=query_id, dataset_name=dataset_name)
        query_emb_batch = query_embedding.unsqueeze(0)  # (1, 384)

        # 2. Budget-Conditioned Feature Modulation
        budget_tensor = torch.tensor([[budget_total]], device=self.device, dtype=query_emb_batch.dtype)
        conditioned_emb = self.budget_conditioner(query_emb_batch, budget_tensor)  # (1, 384)

        # 3. Task classification (always use router's domain, not classifier output)
        # The task classifier is still called for loss computation during training
        _, task_probs, query_context = self.task_classifier(conditioned_emb, self._cached_tasks_emb)

        # Always use the router's configured domain (not task classifier output)
        task_name = self.role_domain
        tasks_role_emb_list = [self._task_role_emb[task_name]]
        tasks_role_list = [self._task_role_database[task_name]]

        # 4. Collaboration / topology selection
        selected_collab_idx, collab_log_probs, collab_context, collab_vae_loss = \
            self.collab_determiner(self._cached_collabs_emb, conditioned_emb)
        topology_index = int(selected_collab_idx[0].item())
        topology_name = self._reasoning_profile[topology_index]["Name"]

        # 5. Number of agents
        agent_num_int, agent_num_float, num_vae_loss = self.num_determiner(conditioned_emb)

        # 6. Role allocation
        planner_context = torch.cat([query_context, collab_context], dim=-1)  # (1, 2*hidden)
        selected_roles_idx, role_log_probs, _, role_vae_loss = \
            self.role_allocation(tasks_role_emb_list, planner_context, agent_num_int)

        # Assemble role names
        role_names = [tasks_role_list[0][int(idx.item())]["Name"] for idx in selected_roles_idx[0]]

        # Total planner log_prob and vae_loss
        planner_log_probs = collab_log_probs + role_log_probs  # (1, 1)
        planner_vae_loss = collab_vae_loss + num_vae_loss + role_vae_loss

        return {
            "query_embedding": query_embedding,
            "topology_name": topology_name,
            "topology_index": topology_index,
            "role_names": role_names,
            "role_set": role_names,
            "role_set_name": "-".join(role_names),
            "agent_count": int(agent_num_int[0].item()),
            "agent_num_float": agent_num_float,
            "task_name": task_name,
            "task_probs": task_probs,
            "planner_log_probs": planner_log_probs,
            "planner_vae_loss": planner_vae_loss,
            "budget_total": budget_total,
            "chosen_role_indices": [int(idx.item()) for idx in selected_roles_idx[0]],
        }

    # ------------------------------------------------------------------
    # Planner: re-evaluate given actions (for training)
    # ------------------------------------------------------------------

    def evaluate_plan(
        self,
        query_embedding: torch.Tensor,
        budget_total: float,
        chosen_topology_idx: int,
        chosen_role_indices: List[int],
        agent_count: int,
    ) -> Dict[str, torch.Tensor]:
        """Re-compute planner forward pass, evaluate log_prob of given actions.

        Creates a fresh computation graph referencing current weights so that
        ``loss.backward()`` never encounters stale weight versions.
        """
        self._ensure_planner_caches()

        query_emb_batch = query_embedding.unsqueeze(0)
        budget_tensor = torch.tensor(
            [[budget_total]], device=self.device, dtype=query_emb_batch.dtype,
        )
        conditioned_emb = self.budget_conditioner(query_emb_batch, budget_tensor)

        # Task classification
        _, task_probs, query_context = self.task_classifier(
            conditioned_emb, self._cached_tasks_emb,
        )

        # Collab — evaluate given topology index
        collab_log_prob, collab_embedding, collab_vae_loss = (
            self.collab_determiner.evaluate(
                self._cached_collabs_emb, conditioned_emb, chosen_topology_idx,
            )
        )

        # Num determiner (deterministic, only need vae_loss)
        _, _, num_vae_loss = self.num_determiner(conditioned_emb)

        # Role allocation — evaluate given role indices
        task_name = self.role_domain
        tasks_role_emb_list = [self._task_role_emb[task_name]]
        planner_context = torch.cat([query_context, collab_embedding], dim=-1)
        agent_num_int_tensor = torch.tensor([agent_count], device=self.device)
        role_log_prob, role_vae_loss = self.role_allocation.evaluate(
            tasks_role_emb_list, planner_context, agent_num_int_tensor,
            [chosen_role_indices],
        )

        planner_log_probs = collab_log_prob + role_log_prob
        planner_vae_loss = collab_vae_loss + num_vae_loss + role_vae_loss

        return {
            "planner_log_probs": planner_log_probs,
            "planner_vae_loss": planner_vae_loss,
            "task_probs": task_probs,
        }

    # ------------------------------------------------------------------
    # Executor: per-node (model, strategy) selection
    # ------------------------------------------------------------------

    def assemble_executor_state(
        self,
        query_embedding: torch.Tensor,
        role_embedding: torch.Tensor,
        budget_remaining: float,
        system_state_vector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        budget_tensor = torch.tensor([budget_remaining], device=self.device, dtype=query_embedding.dtype)
        state_tensor = torch.cat([query_embedding, role_embedding, budget_tensor, system_state_vector], dim=0)
        return {"state": state_tensor, "budget_remaining": budget_tensor}

    def get_executor_action(
        self, executor_state: Union[torch.Tensor, Dict[str, torch.Tensor]], deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        state_tensor = self._as_state_tensor(executor_state)
        hidden = self.executor_backbone(state_tensor)
        model_probs = self.model_head(hidden).softmax(dim=-1)
        strategy_probs = self.strategy_head(hidden).softmax(dim=-1)

        if deterministic:
            model_idx = model_probs.argmax(dim=-1)
            strategy_idx = strategy_probs.argmax(dim=-1)
        else:
            model_idx = torch.distributions.Categorical(model_probs).sample()
            strategy_idx = torch.distributions.Categorical(strategy_probs).sample()

        return {
            "model_index": model_idx,
            "strategy_index": strategy_idx,
            "model_probs": model_probs,
            "strategy_probs": strategy_probs,
            "value": self.value_head(hidden).squeeze(-1),
        }

    def get_system_state_vector(self, dtype: torch.dtype, query_text: str = "", role_name: str = "") -> torch.Tensor:
        """Build ``[L_hat_σ1, ..., L_hat_σS, n_run, n_wait]`` per model."""
        values: List[float] = []
        prompt_len = len(query_text.split()) * 1.3 if query_text else 0.0
        effective_role = role_name or "ProgrammingExpert"
        for model_name in self.models:
            snap = model_metrics.get(model_name, {})
            n_run = float(snap.get("num_requests_running", 0.0))
            n_wait = float(snap.get("num_requests_waiting", 0.0))
            kv_cache = float(snap.get("kv_cache_usage_perc", 0.0))
            avg_tpot = float(snap.get("itl_avg", 0.0))
            avg_ttft = float(snap.get("ttft_avg", 0.0))
            avg_queue = float(snap.get("queue_avg", 0.0))
            avg_inference = float(snap.get("inference_avg", 0.0))
            for strategy_name in self.strategies:
                l_hat = 0.0
                if self.latency_predictor is not None and self.length_predictor is not None and query_text:
                    try:
                        ttft_hat, tpot_hat = self.latency_predictor.predict_latency(
                            prompt_len=prompt_len, waiting_queue=n_wait, running_queue=n_run,
                            kv_cache_usage=kv_cache, avg_tpot=avg_tpot, avg_ttft=avg_ttft,
                            avg_queue=avg_queue, avg_inference=avg_inference,
                            model_name=model_name, role_name=effective_role, strategy_name=strategy_name,
                        )
                        length_hat = self.length_predictor.predict_length(
                            query_text, model_name=model_name, role_name=effective_role, strategy_name=strategy_name,
                        )
                        l_hat = ttft_hat + tpot_hat * length_hat
                    except Exception as exc:
                        logger.trace("[Router] Predictor error for {} / {}: {}", model_name, strategy_name, exc)
                values.append(l_hat)
            values.extend([n_run, n_wait])
        return torch.tensor(values, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def compute_executor_reward(
        self,
        semantic_quality: Union[float, torch.Tensor],
        latency: Union[float, torch.Tensor],
        budget_remaining: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """CMDP reward per executor step: quality − λ · (step_latency / budget_remaining)."""
        quality = self._to_tensor(semantic_quality)
        lat = self._to_tensor(latency).to(self.device)
        bud = self._to_tensor(budget_remaining).clamp_min(1.0)
        return quality - F.softplus(self.lagrange_multiplier) * (lat / bud)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _as_state_tensor(self, state: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        t = state["state"] if isinstance(state, dict) else state
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def _to_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else torch.tensor([value], device=self.device)

    def load_mas_checkpoint(self, path: str) -> None:
        """Load pretrained MAS Router weights into shared planner modules.

        Transfers: task_classifier, collab_determiner, num_determiner,
        role_allocation.  Skips MAS-only modules (text_encoder, llm_router)
        and leaves InfraMind-only modules (budget_conditioner, executor, etc.)
        at their random init.
        """
        _SHARED_PREFIXES = (
            "task_classifier.",
            "collab_determiner.",
            "num_determiner.",
            "role_allocation.",
        )
        mas_state = torch.load(path, map_location=self.device)
        # MAS saves plain state_dict via torch.save(router.state_dict(), ...)
        if "router_state_dict" in mas_state:
            mas_state = mas_state["router_state_dict"]

        matched, skipped = 0, 0
        own_state = self.state_dict()
        for key, value in mas_state.items():
            if not key.startswith(_SHARED_PREFIXES):
                skipped += 1
                continue
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                matched += 1
            else:
                logger.warning("[MAS→InfraMind] Shape mismatch or missing key: {}", key)
                skipped += 1

        self.load_state_dict(own_state)
        logger.info(
            "[MAS→InfraMind] Loaded {} params from MAS checkpoint, skipped {} (path={})",
            matched, skipped, path,
        )

    def planner_parameters(self):
        """All parameters that belong to the planner (for optimizer)."""
        params: List[torch.Tensor] = []
        for module in [self.budget_conditioner, self.task_classifier,
                       self.collab_determiner, self.num_determiner, self.role_allocation]:
            params.extend(module.parameters())
        return params

    def executor_parameters(self):
        """All parameters that belong to the executor (for optimizer)."""
        params: List[torch.Tensor] = []
        for module in [self.executor_backbone, self.model_head,
                       self.strategy_head, self.value_head]:
            params.extend(module.parameters())
        return params


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _load_models_from_profile() -> List[str]:
    profile_path = Path(__file__).resolve().parents[1] / "LLM" / "llm_profile_full.json"
    if not profile_path.is_file():
        return []
    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [m["Name"] for m in data.get("models", []) if isinstance(m, dict) and "Name" in m]
    except (OSError, json.JSONDecodeError):
        return []


def _load_role_profiles(domain: str) -> Dict[str, Dict[str, object]]:
    roles_dir = Path(__file__).resolve().parents[2] / "MAR" / "Roles" / domain
    if not roles_dir.is_dir():
        raise RuntimeError(f"Role domain not found: {roles_dir}")
    profiles: Dict[str, Dict[str, object]] = {}
    for path in sorted(roles_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        profiles[str(profile.get("Name") or path.stem)] = profile
    if not profiles:
        raise RuntimeError(f"No role profiles found under {roles_dir}")
    return profiles


def _load_reasoning_profile() -> List[Dict[str, object]]:
    profile_path = Path(__file__).resolve().parents[2] / "MAR" / "Agent" / "reasoning_profile.py"
    if not profile_path.is_file():
        raise RuntimeError(f"Reasoning profile not found: {profile_path}")
    data = runpy.run_path(str(profile_path))
    profile = data.get("reasoning_profile")
    if not isinstance(profile, list):
        raise RuntimeError("Reasoning profile file did not define a list named reasoning_profile.")
    return profile


def _load_tasks_profile() -> List[Dict[str, str]]:
    profile_path = Path(__file__).resolve().parents[2] / "MAR" / "Prompts" / "tasks_profile.py"
    if not profile_path.is_file():
        raise RuntimeError(f"Tasks profile not found: {profile_path}")
    data = runpy.run_path(str(profile_path))
    return data["tasks_profile"]


def _load_predictors(latency_path: str, length_path: str, encoder: "SemanticEncoder", device: torch.device) -> Tuple:
    from MAR.InfraMind.latency_estimator import LatencyEstimatorBundle, load_latency_estimator
    from MAR.InfraMind.length_estimator import LengthEstimatorBundle, load_length_estimator
    lat_path, len_path = Path(latency_path), Path(length_path)
    if not lat_path.is_file():
        raise FileNotFoundError(f"Latency predictor checkpoint not found: {lat_path}")
    if not len_path.is_file():
        raise FileNotFoundError(f"Length predictor checkpoint not found: {len_path}")
    lat_model, lat_meta, _ = load_latency_estimator(lat_path, device=device)
    len_model, len_meta, _ = load_length_estimator(len_path, device=device)
    return (
        LatencyEstimatorBundle(model=lat_model, metadata=lat_meta),
        LengthEstimatorBundle(model=len_model, metadata=len_meta, encoder=encoder),
    )
