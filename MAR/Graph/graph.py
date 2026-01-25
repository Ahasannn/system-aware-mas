import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import time
import threading
import numpy as np
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from MAR.Graph.node import Node
from MAR.Utils.telemetry import GraphTrace, NodeTiming, utc_now_iso, LLMUsageTracker
from MAR.Utils.utils import find_mode
from MAR.Agent.agent_registry import AgentRegistry

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_names: List[str],
                agent_names: List[str],
                decision_method: str,
                reasoning_name: str,
                prompt_file: str,
                runtime_llm_assignment: bool = False,
                latency_budget: Optional[str] = None,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                **kwargs,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_names:List[str] = llm_names
        self.final_llm_name:str = find_mode(llm_names)
        self.agent_names:List[str] = agent_names
        self.runtime_llm_assignment = runtime_llm_assignment
        self.runtime_llm_map: Dict[str, str] = {}
        self.latency_budget = latency_budget
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.final_llm_name, "prompt_file":prompt_file})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.reasoning_name = reasoning_name

        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
                                                 requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name, llm_name, kwargs in zip(self.agent_names, self.llm_names, self.node_kwargs):
            if "Agent" in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = "" if self.runtime_llm_assignment else llm_name
                kwargs["reason_name"] = self.reasoning_name
                kwargs["role"] = agent_name
                if self.latency_budget:
                    kwargs["latency_budget"] = self.latency_budget
                agent_instance = AgentRegistry.get("Agent", **kwargs)
                agent_instance = self.add_node(agent_instance)
                if self.runtime_llm_assignment:
                    self.runtime_llm_map[agent_instance.id] = llm_name

    def _assign_runtime_llm(self, node_id: str) -> None:
        if not self.runtime_llm_assignment:
            return
        llm_name = self.runtime_llm_map.get(node_id)
        if not llm_name:
            return
        node = self.nodes.get(node_id)
        if node is None:
            return
        if hasattr(node, "set_llm"):
            node.set_llm(llm_name)
        else:
            setattr(node, "llm_name", llm_name)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(
        self,
        inputs: Dict[str, str],
        num_rounds: int = 2,
        max_tries: int = 3,
        max_time: int = 100,
        request_timeout: Optional[float] = None,
        trace: Optional[GraphTrace] = None,
    ) -> Tuple[List[Any], Any]:
        if trace is not None:
            trace.start_workflow()

        usage_tracker = LLMUsageTracker.instance()

        def _final_output_text(node: Node) -> str:
            outputs = getattr(node, "outputs", [])
            if isinstance(outputs, list):
                if len(outputs) == 0:
                    return ""
                if len(outputs) == 1:
                    return str(outputs[0])
                return "\n\n".join(str(value) for value in outputs)
            return str(outputs)

        def _safe_role_name(node: Node) -> str:
            role = getattr(node, "role", "")
            if hasattr(role, "role"):
                return str(getattr(role, "role"))
            if isinstance(role, str):
                return role
            return ""

        def _safe_llm_name(node: Node) -> str:
            llm = getattr(node, "llm", None)
            model_name = getattr(llm, "model_name", None)
            if isinstance(model_name, str):
                return model_name
            llm_name = getattr(node, "llm_name", None)
            if isinstance(llm_name, str):
                return llm_name
            return ""

        log_probs = 0
        step_counter = 0
        llm_elapsed_seconds = 0.0
        transitions: Dict[Tuple[int, str], Dict[str, Any]] = {}
        transitions_lock = threading.Lock()
        workflow_error = ""
        try:
            usage_tracker = LLMUsageTracker.instance()
            for round in range(num_rounds):
                log_probs += self.construct_spatial_connection()
                log_probs += self.construct_temporal_connection(round)

                in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
                zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
                wave_idx = 0

                while zero_in_degree_queue:
                    current_wave_ids = zero_in_degree_queue
                    zero_in_degree_queue = []
                    wave_step_indices = {
                        node_id: step_counter + offset for offset, node_id in enumerate(current_wave_ids)
                    }
                    step_counter += len(current_wave_ids)

                    for node_id in current_wave_ids:
                        node = self.nodes[node_id]
                        transitions[(round, node_id)] = {
                            "step_index": wave_step_indices[node_id],
                            "round_index": round,
                            "wave_index": wave_idx,
                            "node_id": node_id,
                            "role_name": _safe_role_name(node),
                            "llm_name": "",
                            "latency_seconds": 0.0,
                            "llm_elapsed_seconds": float(llm_elapsed_seconds),
                        }

                    def run_node(node_id: str) -> None:
                        node = self.nodes[node_id]
                        role_name = _safe_role_name(node)
                        llm_name = _safe_llm_name(node)
                        ts_start = utc_now_iso()
                        start_perf = time.perf_counter()
                        usage_key = f"{self.id}:{round}:{node_id}"
                        context_token = usage_tracker.set_context(usage_key)
                        usage_tracker.clear(usage_key)
                        usage = {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0}
                        tries = 0
                        success = False
                        error_msg = ""
                        timed_out = False
                        try:
                            while tries < max_tries:
                                tries += 1
                                try:
                                    self._assign_runtime_llm(node_id)
                                    self.nodes[node_id].execute(
                                        inputs,
                                        request_timeout=request_timeout,
                                    )  # output is saved in the node.outputs
                                    success = True
                                    break
                                except Exception as e:
                                    error_msg = str(e)
                                    print(f"[DEBUG] Graph {self.id}, Node {node_id}, Attempt {tries}/{max_tries} failed")
                                    print(f"[DEBUG] Exception type: {type(e).__name__}")
                                    print(f"[DEBUG] Exception message: {error_msg}")
                                    if isinstance(e, TimeoutError):
                                        timed_out = True
                                        break
                                    if tries >= max_tries:
                                        logger.error(f"Graph {self.id}: Node {node_id} failed after {max_tries} attempts: {error_msg}")
                                    else:
                                        print(f"[DEBUG] Retrying node {node_id} (attempt {tries + 1}/{max_tries})...")
                        finally:
                            usage = usage_tracker.consume(usage_key)
                            usage_tracker.reset_context(context_token)
                        ts_end = utc_now_iso()
                        duration_sec = time.perf_counter() - start_perf
                        node = self.nodes[node_id]
                        with transitions_lock:
                            transition = transitions.get((round, node_id))
                            if transition is not None:
                                transition["llm_name"] = _safe_llm_name(node)
                                transition["latency_seconds"] = duration_sec
                        if trace is not None:
                            output_text = ""
                            trace.record_node_event(
                                NodeTiming(
                                    round_idx=round,
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
                                    cost_delta=float(usage.get("cost", 0.0)),
                                    prompt_tokens=int(usage.get("prompt_tokens", 0.0)),
                                    completion_tokens=int(usage.get("completion_tokens", 0.0)),
                                    output_text=output_text,
                                )
                            )
                        if timed_out:
                            raise TimeoutError(error_msg or "LLM request timed out")

                    max_workers = len(current_wave_ids)
                    if max_workers <= 1:
                        for node_id in current_wave_ids:
                            run_node(node_id)
                    else:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [executor.submit(run_node, node_id) for node_id in current_wave_ids]
                            for future in as_completed(futures):
                                future.result()

                    for node_id in current_wave_ids:
                        for successor in self.nodes[node_id].spatial_successors:
                            if successor.id not in self.nodes.keys():
                                continue
                            in_degree[successor.id] -= 1
                            if in_degree[successor.id] == 0:
                                zero_in_degree_queue.append(successor.id)
                    wave_max_latency = 0.0
                    for node_id in current_wave_ids:
                        transition = transitions.get((round, node_id))
                        if transition:
                            latency = float(transition.get("latency_seconds", 0.0))
                            if latency > wave_max_latency:
                                wave_max_latency = latency
                    llm_elapsed_seconds += wave_max_latency
                    wave_idx += 1
                self.update_memory()

            self.connect_decision_node()
            ts_start = utc_now_iso()
            start_perf = time.perf_counter()
            decision_node_id = getattr(self.decision_node, "id", "")
            usage_key = f"{self.id}:{num_rounds}:{decision_node_id}"
            context_token = usage_tracker.set_context(usage_key)
            usage_tracker.clear(usage_key)
            usage = {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0}
            decision_success = False
            decision_error = ""
            try:
                self.decision_node.execute(inputs, request_timeout=request_timeout)
                decision_success = True
            except Exception as e:
                decision_error = str(e)
                raise
            finally:
                usage = usage_tracker.consume(usage_key)
                usage_tracker.reset_context(context_token)
                ts_end = utc_now_iso()
                duration_sec = time.perf_counter() - start_perf
                if trace is not None:
                    output_text = _final_output_text(self.decision_node)
                    trace.record_node_event(
                        NodeTiming(
                            round_idx=num_rounds,
                            node_id=getattr(self.decision_node, "id", ""),
                            node_name=self.decision_node.node_name,
                            role_name="FinalDecision",
                            llm_name=_safe_llm_name(self.decision_node),
                            is_decision_node=True,
                            attempts=1,
                            success=decision_success,
                            error=decision_error,
                            ts_start=ts_start,
                            ts_end=ts_end,
                            duration_sec=duration_sec,
                            cost_delta=float(usage.get("cost", 0.0)),
                            prompt_tokens=int(usage.get("prompt_tokens", 0.0)),
                            completion_tokens=int(usage.get("completion_tokens", 0.0)),
                            output_text=output_text,
                        )
                    )

            final_answers = self.decision_node.outputs
            if len(final_answers) == 0:
                final_answers.append("No answer of the decision node")

            return final_answers, log_probs
        except Exception as e:
            workflow_error = str(e)
            raise
        finally:
            if trace is not None:
                trace.end_workflow(success=(workflow_error == ""), error=workflow_error)
            ordered_transitions = sorted(transitions.values(), key=lambda item: item["step_index"])
            setattr(self, "last_llm_elapsed_seconds", float(llm_elapsed_seconds))
            setattr(self, "last_transitions", ordered_transitions)

    async def arun(
        self,
        input: Dict[str, str],
        num_rounds: int = 3,
        max_tries: int = 3,
        max_time: int = 600,
        trace: Optional[GraphTrace] = None,
    ) -> Tuple[List[Any], Any]:
        if trace is not None:
            trace.start_workflow()

        usage_tracker = LLMUsageTracker.instance()

        def _final_output_text(node: Node) -> str:
            outputs = getattr(node, "outputs", [])
            if isinstance(outputs, list):
                if len(outputs) == 0:
                    return ""
                if len(outputs) == 1:
                    return str(outputs[0])
                return "\n\n".join(str(value) for value in outputs)
            return str(outputs)

        def _safe_role_name(node: Node) -> str:
            role = getattr(node, "role", "")
            if hasattr(role, "role"):
                return str(getattr(role, "role"))
            if isinstance(role, str):
                return role
            return ""

        def _safe_llm_name(node: Node) -> str:
            llm = getattr(node, "llm", None)
            model_name = getattr(llm, "model_name", None)
            if isinstance(model_name, str):
                return model_name
            llm_name = getattr(node, "llm_name", None)
            if isinstance(llm_name, str):
                return llm_name
            return ""

        log_probs = 0
        workflow_error = ""
        try:
            for round in range(num_rounds):
                log_probs += self.construct_spatial_connection()
                log_probs += self.construct_temporal_connection(round)

                in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
                zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

                while zero_in_degree_queue:
                    current_node_id = zero_in_degree_queue.pop(0)
                    ts_start = utc_now_iso()
                    start_perf = time.perf_counter()
                    usage_key = f"{self.id}:{round}:{current_node_id}"
                    context_token = usage_tracker.set_context(usage_key)
                    usage_tracker.clear(usage_key)
                    usage = {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0}
                    tries = 0
                    success = False
                    error_msg = ""
                    try:
                        while tries < max_tries:
                            tries += 1
                            try:
                                self._assign_runtime_llm(current_node_id)
                                await asyncio.wait_for(
                                    self.nodes[current_node_id].async_execute(input),
                                    timeout=max_time,
                                )  # output is saved in the node.outputs
                                success = True
                                break
                            except Exception as e:
                                error_msg = str(e)
                                print(f"Error during execution of node {current_node_id}: {e}")
                    finally:
                        usage = usage_tracker.consume(usage_key)
                        usage_tracker.reset_context(context_token)

                    ts_end = utc_now_iso()
                    duration_sec = time.perf_counter() - start_perf
                    if trace is not None:
                        node = self.nodes[current_node_id]
                        output_text = ""
                        trace.record_node_event(
                            NodeTiming(
                                round_idx=round,
                                node_id=current_node_id,
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
                                cost_delta=float(usage.get("cost", 0.0)),
                                prompt_tokens=int(usage.get("prompt_tokens", 0.0)),
                                completion_tokens=int(usage.get("completion_tokens", 0.0)),
                                output_text=output_text,
                            )
                        )

                    for successor in self.nodes[current_node_id].spatial_successors:
                        if successor.id not in self.nodes.keys():
                            continue
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            zero_in_degree_queue.append(successor.id)

                self.update_memory()

            self.connect_decision_node()
            ts_start = utc_now_iso()
            start_perf = time.perf_counter()
            decision_node_id = getattr(self.decision_node, "id", "")
            usage_key = f"{self.id}:{num_rounds}:{decision_node_id}"
            context_token = usage_tracker.set_context(usage_key)
            usage_tracker.clear(usage_key)
            usage = {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0}
            decision_success = False
            decision_error = ""
            try:
                await self.decision_node.async_execute(input)
                decision_success = True
            except Exception as e:
                decision_error = str(e)
                raise
            finally:
                usage = usage_tracker.consume(usage_key)
                usage_tracker.reset_context(context_token)
            ts_end = utc_now_iso()
            duration_sec = time.perf_counter() - start_perf
            if trace is not None:
                output_text = _final_output_text(self.decision_node)
                trace.record_node_event(
                    NodeTiming(
                        round_idx=num_rounds,
                        node_id=getattr(self.decision_node, "id", ""),
                        node_name=self.decision_node.node_name,
                            role_name="FinalDecision",
                            llm_name=_safe_llm_name(self.decision_node),
                            is_decision_node=True,
                            attempts=1,
                            success=decision_success,
                            error=decision_error,
                            ts_start=ts_start,
                        ts_end=ts_end,
                        duration_sec=duration_sec,
                        cost_delta=float(usage.get("cost", 0.0)),
                        prompt_tokens=int(usage.get("prompt_tokens", 0.0)),
                        completion_tokens=int(usage.get("completion_tokens", 0.0)),
                        output_text=output_text,
                    )
                )

            final_answers = self.decision_node.outputs
            if len(final_answers) == 0:
                final_answers.append("No answer of the decision node")
            return final_answers, log_probs
        except Exception as e:
            workflow_error = str(e)
            raise
        finally:
            if trace is not None:
                trace.end_workflow(success=(workflow_error == ""), error=workflow_error)
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
    
    def list_nodes(self):
        profile = []
        for i, node_id in enumerate(self.nodes):
            profile.append({'id': node_id, 'role': self.nodes[node_id].role.role, 'llm_name': self.nodes[node_id].llm.model_name})
        return profile
