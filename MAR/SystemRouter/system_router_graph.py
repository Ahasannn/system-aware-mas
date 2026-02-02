import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from MAR.Agent.agent_registry import AgentRegistry
from MAR.Graph.graph import Graph
from MAR.Utils.telemetry import GraphTrace, NodeTiming, utc_now_iso, LLMUsageTracker
from MAR.Utils.utils import find_mode
from MAR.LLM.llm_registry import LLMRegistry
from MAR.SystemRouter.system_router_agent import SystemRouterAgent
from MAR.SystemRouter.metrics_watcher import model_metrics


class SystemRouterGraph(Graph):
    def init_nodes(self):
        for agent_name, llm_name, kwargs in zip(self.agent_names, self.llm_names, self.node_kwargs):
            if "SystemRouterAgent" in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = "" if self.runtime_llm_assignment else llm_name
                kwargs["reason_name"] = self.reasoning_name
                kwargs["role"] = agent_name
                if self.latency_budget:
                    kwargs["latency_budget"] = self.latency_budget
                agent_instance = AgentRegistry.get("SystemRouterAgent", **kwargs)
                agent_instance = self.add_node(agent_instance)
                if self.runtime_llm_assignment:
                    self.runtime_llm_map[agent_instance.id] = llm_name

    def run_with_policy(
        self,
        inputs: Dict[str, str],
        router: Any,
        query_embedding: torch.Tensor,
        budget_total: float,
        num_rounds: int = 1,
        deterministic: bool = False,
        max_tries: int = 3,
        router_lock: Optional[threading.Lock] = None,
        system_state_vector: Optional[torch.Tensor] = None,
        trace: Optional[GraphTrace] = None,
    ) -> Dict[str, Any]:
        if trace is not None:
            trace.start_workflow()

        usage_tracker = LLMUsageTracker.instance()
        workflow_error = ""
        log_probs = 0
        step_counter = 0
        llm_elapsed_seconds = 0.0

        transitions: Dict[Tuple[int, str], Dict[str, Any]] = {}
        token_tally = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def _final_output_text(node) -> str:
            outputs = getattr(node, "outputs", [])
            if isinstance(outputs, list):
                if len(outputs) == 0:
                    return ""
                if len(outputs) == 1:
                    return str(outputs[0])
                return "\n\n".join(str(value) for value in outputs)
            return str(outputs)

        def _safe_role_name(node) -> str:
            role = getattr(node, "role", "")
            if hasattr(role, "role"):
                return str(getattr(role, "role"))
            if isinstance(role, str):
                return role
            return ""

        def _safe_llm_name(node) -> str:
            llm = getattr(node, "llm", None)
            model_name = getattr(llm, "model_name", None)
            if isinstance(model_name, str):
                return model_name
            llm_name = getattr(node, "llm_name", None)
            if isinstance(llm_name, str):
                return llm_name
            return ""

        def _record_token_usage(usage: Dict[str, float]) -> Dict[str, int]:
            prompt_tokens = int(usage.get("prompt_tokens", 0.0))
            completion_tokens = int(usage.get("completion_tokens", 0.0))
            total_tokens = prompt_tokens + completion_tokens
            token_tally["prompt_tokens"] += prompt_tokens
            token_tally["completion_tokens"] += completion_tokens
            token_tally["total_tokens"] += total_tokens
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        def select_action_single(node_id: str, round_idx: int, level: int, path_latency: float) -> None:
            nonlocal step_counter
            node = self.nodes[node_id]
            role_name = _safe_role_name(node)
            budget_remaining = max(budget_total - path_latency, 0.0)
            if router_lock:
                with router_lock:
                    role_embedding = router.encode_role(role_name)
                    state_vector = system_state_vector
                    if state_vector is None:
                        state_vector = router.get_system_state_vector(query_embedding.dtype)
                    exec_state = router.assemble_executor_state(
                        query_embedding, role_embedding, budget_remaining, state_vector
                    )
                    exec_action = router.get_executor_action(exec_state, deterministic=deterministic)
            else:
                role_embedding = router.encode_role(role_name)
                state_vector = system_state_vector
                if state_vector is None:
                    state_vector = router.get_system_state_vector(query_embedding.dtype)
                exec_state = router.assemble_executor_state(
                    query_embedding, role_embedding, budget_remaining, state_vector
                )
                exec_action = router.get_executor_action(exec_state, deterministic=deterministic)
            model_idx = int(exec_action["model_index"].item())
            strategy_idx = int(exec_action["strategy_index"].item())
            model_name = router.models[model_idx]
            strategy_name = router.strategies[strategy_idx]

            if hasattr(node, "set_llm"):
                node.set_llm(model_name)
            else:
                setattr(node, "llm_name", model_name)
            if hasattr(node, "set_strategy"):
                node.set_strategy(strategy_name)

            predecessors = "|".join(
                p.id for p in node.spatial_predecessors if p.id in self.nodes
            )
            successors = "|".join(
                s.id for s in node.spatial_successors if s.id in self.nodes
            )
            transitions[(round_idx, node_id)] = {
                "state": exec_state,
                "action": exec_action,
                "role": role_name,
                "step_index": step_counter,
                "round_index": round_idx,
                "dep_level": level,
                "node_id": node_id,
                "model": model_name,
                "strategy": strategy_name,
                "budget_remaining": float(budget_remaining),
                "llm_elapsed_seconds": float(path_latency),
                "spatial_predecessors": predecessors,
                "spatial_successors": successors,
            }
            step_counter += 1

        def run_node(node_id: str, round_idx: int) -> None:
            node = self.nodes[node_id]
            transition = transitions[(round_idx, node_id)]
            ts_start = utc_now_iso()
            start_perf = time.perf_counter()
            usage_key = f"{self.id}:{round_idx}:{node_id}"
            context_token = usage_tracker.set_context(usage_key)
            usage_tracker.clear(usage_key)
            model_name = transition.get("model", "")
            snap = model_metrics.get(model_name, {})
            if not snap:
                from loguru import logger
                logger.trace("[Graph] No metrics found for model '{}'. Available: {}", model_name, list(model_metrics.keys()))
            transition["llm_running"] = snap.get("num_requests_running", 0)
            transition["llm_waiting"] = snap.get("num_requests_waiting", 0)
            transition["llm_kv_cache_usage"] = snap.get("kv_cache_usage_perc", 0.0)
            transition["llm_ttft_avg"] = snap.get("ttft_avg", 0.0)
            transition["llm_itl_avg"] = snap.get("itl_avg", 0.0)
            transition["llm_e2e_avg"] = snap.get("e2e_avg", 0.0)
            transition["llm_queue_avg"] = snap.get("queue_avg", 0.0)
            transition["llm_inference_avg"] = snap.get("inference_avg", 0.0)
            tries = 0
            success = False
            error_msg = ""
            try:
                while tries < max_tries:
                    tries += 1
                    try:
                        node.execute(inputs, max_tokens=getattr(self, "max_tokens", None))
                        success = True
                        break
                    except Exception as e:
                        error_msg = str(e)
            finally:
                usage = usage_tracker.consume(usage_key)
                usage_tracker.reset_context(context_token)

            ts_end = utc_now_iso()
            duration_sec = time.perf_counter() - start_perf
            response_text = _final_output_text(node)
            prompt_base = getattr(node, "prompt_base", "")
            token_counts = _record_token_usage(usage)
            observed_ttft = float(getattr(node, "last_ttft", 0.0))
            observed_tpot = float(getattr(node, "last_tpot", 0.0))

            transition["latency_seconds"] = duration_sec
            transition["response"] = response_text
            transition["prompt_base"] = prompt_base
            transition["token_counts"] = token_counts
            transition["observed_ttft"] = observed_ttft
            transition["observed_tpot"] = observed_tpot
            transition["success"] = bool(success)
            transition["error"] = error_msg

            if trace is not None:
                trace.record_node_event(
                    NodeTiming(
                        round_idx=round_idx,
                        node_id=node_id,
                        node_name=node.node_name,
                        role_name=_safe_role_name(node),
                        llm_name=_safe_llm_name(node),
                        is_decision_node=False,
                        attempts=tries,
                        success=success,
                        error=error_msg,
                        ts_start=ts_start,
                        ts_end=ts_end,
                        duration_sec=duration_sec,
                        cost_delta=0.0,
                        prompt_tokens=token_counts["prompt_tokens"],
                        completion_tokens=token_counts["completion_tokens"],
                        output_text=response_text,
                    )
                )

            if not success:
                raise RuntimeError(f"Agent {node_id} failed after {tries} tries: {error_msg}")

        try:
            for round_idx in range(num_rounds):
                log_probs += self.construct_spatial_connection()
                log_probs += self.construct_temporal_connection(round_idx)

                # Topological sort via BFS to get execution order and dependency levels
                in_degree = {
                    node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()
                }
                queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
                exec_order: List[str] = []
                dep_level_map: Dict[str, int] = {nid: 0 for nid in queue}
                bfs_queue = list(queue)
                while bfs_queue:
                    nid = bfs_queue.pop(0)
                    exec_order.append(nid)
                    for successor in self.nodes[nid].spatial_successors:
                        if successor.id not in self.nodes:
                            continue
                        new_level = dep_level_map[nid] + 1
                        dep_level_map[successor.id] = max(dep_level_map.get(successor.id, 0), new_level)
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            bfs_queue.append(successor.id)

                # Per-node cumulative path latency (max ancestor path to this node)
                node_path_latency: Dict[str, float] = {}

                # Execute nodes sequentially in topological order
                for nid in exec_order:
                    node = self.nodes[nid]
                    # Compute path latency: max of (pred path + pred own latency)
                    preds = [p for p in node.spatial_predecessors if p.id in self.nodes]
                    if not preds:
                        node_path_latency[nid] = llm_elapsed_seconds  # carry over from previous rounds
                    else:
                        node_path_latency[nid] = max(
                            node_path_latency[p.id]
                            + float(transitions[(round_idx, p.id)].get("latency_seconds", 0.0))
                            for p in preds
                        )

                    # Select action with exact path-based budget
                    select_action_single(nid, round_idx, dep_level_map[nid], node_path_latency[nid])

                    # Execute the node
                    run_node(nid, round_idx)

                # Update llm_elapsed_seconds to the critical path through this round
                if exec_order:
                    llm_elapsed_seconds = max(
                        node_path_latency[nid]
                        + float(transitions[(round_idx, nid)].get("latency_seconds", 0.0))
                        for nid in exec_order
                    )

                self.update_memory()

            self.connect_decision_node()
            decision_node = self.decision_node
            selected_models = [t["model"] for t in transitions.values() if t.get("model")]
            decision_llm_name = find_mode(selected_models) if selected_models else self.final_llm_name
            if decision_llm_name:
                setattr(decision_node, "llm_name", decision_llm_name)
                setattr(decision_node, "llm", LLMRegistry.get(decision_llm_name))

            ts_start = utc_now_iso()
            start_perf = time.perf_counter()
            decision_node_id = getattr(decision_node, "id", "")
            usage_key = f"{self.id}:{num_rounds}:{decision_node_id}"
            context_token = usage_tracker.set_context(usage_key)
            usage_tracker.clear(usage_key)
            decision_success = False
            decision_error = ""
            try:
                decision_node.execute(inputs, max_tokens=getattr(self, "max_tokens", None))
                decision_success = True
            except Exception as e:
                decision_error = str(e)
                raise
            finally:
                usage = usage_tracker.consume(usage_key)
                usage_tracker.reset_context(context_token)

            ts_end = utc_now_iso()
            duration_sec = time.perf_counter() - start_perf
            decision_tokens = _record_token_usage(usage)
            decision_text = _final_output_text(decision_node)
            if trace is not None:
                trace.record_node_event(
                    NodeTiming(
                        round_idx=num_rounds,
                        node_id=decision_node_id,
                        node_name=decision_node.node_name,
                        role_name="FinalDecision",
                        llm_name=_safe_llm_name(decision_node),
                        is_decision_node=True,
                        attempts=1,
                        success=decision_success,
                        error=decision_error,
                        ts_start=ts_start,
                        ts_end=ts_end,
                        duration_sec=duration_sec,
                        cost_delta=0.0,
                        prompt_tokens=decision_tokens["prompt_tokens"],
                        completion_tokens=decision_tokens["completion_tokens"],
                        output_text=decision_text,
                    )
                )

            # Workflow latency = critical path through the DAG (already computed as llm_elapsed_seconds)
            workflow_latency = float(llm_elapsed_seconds)
            ordered_transitions = sorted(transitions.values(), key=lambda item: item["step_index"])
            return {
                "final_response": decision_text,
                "executor_transitions": ordered_transitions,
                "workflow_latency_seconds": workflow_latency,
                "llm_elapsed_seconds": float(llm_elapsed_seconds),
                "token_counts": token_tally,
                "log_probs": log_probs,
            }
        except Exception as e:
            workflow_error = str(e)
            raise
        finally:
            if trace is not None:
                trace.end_workflow(success=(workflow_error == ""), error=workflow_error)
