from typing import Dict, List
import json
import os
import time
from loguru import logger

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_profile_full import get_model_max_context_len, get_model_max_output_tokens
from MAR.LLM.llm_registry import LLMRegistry
from MAR.LLM.llm import LLM
from MAR.LLM.gpt_chat import _resolve_api_key, _resolve_base_url, _get_shared_sync_client, _normalize_request_timeout
from MAR.LLM.price import cal_token, truncate_text_for_model, cost_count
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation import message_aggregation,inner_test
from MAR.Prompts.post_process import post_process
from MAR.Prompts.output_format import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt
from MAR.SystemRouter.prompt_strategies import get_strategy_prompt


def limit_prompt_for_llm(llm_model_name: str, system_prompt: str, user_prompt: str) -> str:
    max_model_len = get_model_max_context_len(llm_model_name)
    try:
        max_out = int(get_model_max_output_tokens(llm_model_name))
    except Exception:
        max_out = 512
    # Reserve space for the model's output plus a small overhead for chat formatting.
    prompt_budget = max(0, max_model_len - max_out - 32)
    system_tokens = cal_token(llm_model_name, system_prompt)
    available = max(0, prompt_budget - system_tokens)
    if available <= 0:
        return ""
    user_tokens = cal_token(llm_model_name, user_prompt)
    if user_tokens <= available:
        return user_prompt
    trimmed = truncate_text_for_model(user_prompt, available, llm_model_name)
    logger.debug(
        "Trimmed prompt for %s from %d to %d tokens (budget %d)",
        llm_model_name,
        user_tokens,
        cal_token(llm_model_name, trimmed),
        available,
    )
    return trimmed


def resolve_max_output_tokens(llm_model_name: str, prompt_messages, default_max_output_tokens: int = 512) -> int:
    """
    Cap completion length so agents don't generate up to the full context window.

    Note: prompts are already trimmed to leave headroom (see `limit_prompt_for_llm`), but the
    output budget wasn't enforced previously.
    """
    try:
        max_model_len = int(get_model_max_context_len(llm_model_name))
    except Exception:
        max_model_len = 4096

    prompt_text = ""
    if isinstance(prompt_messages, list):
        prompt_text = "".join(
            msg.get("content", "") for msg in prompt_messages if isinstance(msg, dict)
        )
    elif isinstance(prompt_messages, str):
        prompt_text = prompt_messages

    try:
        prompt_tokens = int(cal_token(llm_model_name, prompt_text))
    except Exception:
        prompt_tokens = 0

    # Reserve a small overhead for chat formatting / tokenization mismatch.
    available = max(0, max_model_len - prompt_tokens - 32)

    # Never request more tokens than are available, and never request <= 0.
    # (A fixed floor like 16 can violate `available` and trigger server-side 400s.)
    try:
        default_max_output_tokens = int(default_max_output_tokens)
    except Exception:
        default_max_output_tokens = 512
    return max(1, min(default_max_output_tokens, available))


def fit_messages_to_context(
    llm_model_name: str,
    messages: List[Dict[str, str]],
    max_context_len: int,
    max_tokens: int,
    extra_margin: int = 32,
) -> tuple[List[Dict[str, str]], int, int]:
    """
    Ensure prompt + max_tokens fits within the context window by trimming the last user message.
    Returns (messages, max_tokens, prompt_tokens).
    """
    try:
        max_tokens = int(max_tokens)
    except Exception:
        max_tokens = 1
    if max_tokens <= 0:
        max_tokens = 1
    max_context_len = int(max_context_len) if max_context_len else 0
    if max_context_len <= 0 or not isinstance(messages, list) or not messages:
        prompt_tokens = sum(
            cal_token(llm_model_name, msg.get("content", "")) for msg in messages if isinstance(msg, dict)
        )
        return messages, max_tokens, prompt_tokens

    def _count_tokens(msgs: List[Dict[str, str]]) -> int:
        return sum(
            cal_token(llm_model_name, msg.get("content", "")) for msg in msgs if isinstance(msg, dict)
        )

    prompt_tokens = _count_tokens(messages)
    try:
        extra_margin = int(extra_margin)
    except Exception:
        extra_margin = 32
    if extra_margin < 0:
        extra_margin = 0
    reserve_tokens = max_tokens + extra_margin
    if prompt_tokens + reserve_tokens <= max_context_len:
        return messages, max_tokens, prompt_tokens

    # Trim the last user message to fit the budget.
    user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_idx = idx
            break
    if user_idx is None:
        # No user message to trim; fall back to shrinking max_tokens only.
        max_tokens = max(1, max_context_len - prompt_tokens - extra_margin)
        return messages, max_tokens, prompt_tokens

    other_tokens = _count_tokens([m for i, m in enumerate(messages) if i != user_idx])
    budget = max_context_len - reserve_tokens
    available_for_user = max(0, budget - other_tokens)
    user_content = messages[user_idx].get("content", "")
    trimmed_user = truncate_text_for_model(user_content, available_for_user, llm_model_name)

    new_messages = [m if not isinstance(m, dict) else dict(m) for m in messages]
    new_messages[user_idx]["content"] = trimmed_user
    prompt_tokens = _count_tokens(new_messages)

    if prompt_tokens + reserve_tokens > max_context_len:
        max_tokens = max(1, max_context_len - prompt_tokens - extra_margin)
    return new_messages, max_tokens, prompt_tokens


@AgentRegistry.register('Agent')
class Agent(Node):
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        role: str | None = None,
        llm_name: str = "",
        reason_name: str = "",
        latency_budget: str | None = None,
    ):
        super().__init__(id, reason_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.role = RoleRegistry(domain, role)
        self.reason = reason_name
        self.latency_budget = latency_budget

        self.message_aggregation = self.role.get_message_aggregation()
        self.description = self.role.get_description()
        self.output_format = self.role.get_output_format()
        self.post_process = self.role.get_post_process()
        self.post_description = self.role.get_post_description()
        self.post_output_format = self.role.get_post_output_format()

        # Timing metrics for telemetry (like SystemRouterAgent)
        self.last_ttft = 0.0
        self.last_tpot = 0.0

        # Strategy injection (used only for predictor data generation)
        self.strategy_name = ""
        self.strategy_prompt = ""

        # Reflect
        if reason_name == "Reflection" and self.post_output_format == "None":
            self.post_output_format = self.output_format
            self.post_description = "\nReflect on possible errors in the answer above and answer again using the same format. If you think there are no errors in your previous answers that will affect the results, there is no need to correct them.\n"
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str, Dict], temporal_info:Dict[str, Dict], **kwargs):
        query = raw_inputs['query']
        spatial_prompt = message_aggregation(raw_inputs, spatial_info, self.message_aggregation)
        temporal_prompt = message_aggregation(raw_inputs, temporal_info, self.message_aggregation)
        format_prompt = output_format_prompt[self.output_format]
        reason_prompt = reasoning_prompt[self.reason]

        base_system = f"{self.description}\n{reason_prompt}"
        if self.latency_budget:
            base_system += f"\nLatency budget: {self.latency_budget}. Work a bit faster and be concise."
        base_system += f"\nFormat requirements that must be followed:\n{format_prompt}" if format_prompt else ""
        system_prompt = base_system
        if self.strategy_prompt:
            system_prompt = f"{self.strategy_prompt}\n{base_system}"
        user_prompt = f"{query}\n"
        spatial_section = (
            f"At the same time, other agents' outputs are as follows:\n\n{spatial_prompt}"
            if spatial_prompt
            else ""
        )
        temporal_section = (
            f"In the last round of dialogue, other agents' outputs were:\n\n{temporal_prompt}"
            if temporal_prompt
            else ""
        )

        query_block = f"{query}\n"
        prompt_budget = self._prompt_budget()
        system_tokens = cal_token(self.llm.model_name, system_prompt)
        query_tokens = cal_token(self.llm.model_name, query_block)
        history_budget = max(0, prompt_budget - system_tokens - query_tokens)

        def _trim_history_block(block: str, budget: int) -> tuple[str, int]:
            if not block or budget <= 0:
                return "", max(budget, 0)
            tokens = cal_token(self.llm.model_name, block)
            if tokens <= budget:
                return block, budget - tokens
            trimmed = truncate_text_for_model(block, budget, self.llm.model_name)
            return trimmed, 0

        temporal_section, history_budget = _trim_history_block(temporal_section, history_budget)
        spatial_section, history_budget = _trim_history_block(spatial_section, history_budget)

        user_prompt = query_block
        if spatial_section:
            user_prompt += f"\n\n{spatial_section}"
        if temporal_section:
            user_prompt += f"\n\n{temporal_section}"

        user_prompt = self._enforce_budget(system_prompt, user_prompt, prompt_budget)
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]

    def _prompt_budget(self) -> int:
        max_model_len = get_model_max_context_len(self.llm.model_name)
        reserved_response_tokens = 512
        return max(128, max_model_len - reserved_response_tokens)

    def _enforce_budget(
        self,
        system_prompt: str,
        user_prompt: str,
        prompt_budget: int | None = None,
    ) -> str:
        prompt_budget = prompt_budget if prompt_budget is not None else self._prompt_budget()
        if prompt_budget <= 0:
            return ""
        system_tokens = cal_token(self.llm.model_name, system_prompt)
        available = max(0, prompt_budget - system_tokens)
        if available <= 0:
            return ""
        user_tokens = cal_token(self.llm.model_name, user_prompt)
        if user_tokens <= available:
            return user_prompt
        trimmed = truncate_text_for_model(user_prompt, available, self.llm.model_name)
        logger.debug(
            "Trimmed %s user prompt from %d to %d tokens (budget %d)",
            self.llm.model_name,
            user_tokens,
            cal_token(self.llm.model_name, trimmed),
            available,
        )
        return trimmed

    def set_strategy(self, strategy: str) -> None:
        """Set prompt strategy for predictor data generation."""
        self.strategy_name = strategy
        self.strategy_prompt = get_strategy_prompt(strategy) if strategy else ""

    def set_llm(self, llm_name: str) -> None:
        if not llm_name:
            return
        if llm_name == self.llm_name:
            return
        self.llm_name = llm_name
        self.llm = LLMRegistry.get(llm_name)

    def _call_llm_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
        request_timeout: float | None = None,
    ) -> str:
        """Call LLM with streaming to capture TTFT and TPOT metrics."""
        timeout = _normalize_request_timeout(request_timeout)
        base_client = _get_shared_sync_client(
            base_url=_resolve_base_url(self.llm.model_name),
            api_key=_resolve_api_key(),
        )
        client = base_client.with_options(timeout=timeout)
        if max_tokens is None:
            max_tokens = LLM.DEFAULT_MAX_TOKENS
        else:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = 1
            if max_tokens > LLM.DEFAULT_MAX_TOKENS:
                max_tokens = LLM.DEFAULT_MAX_TOKENS

        max_context_len = get_model_max_context_len(self.llm.model_name)
        messages, max_tokens, _ = fit_messages_to_context(
            self.llm.model_name, messages, max_context_len, max_tokens
        )

        start = time.perf_counter()
        first_token_time = None
        parts: List[str] = []
        def _create_stream(msgs, max_out):
            return client.chat.completions.create(
                model=self.llm.model_name,
                messages=msgs,
                stream=True,
                max_tokens=max_out,
                temperature=LLM.DEFAULT_TEMPERATURE,
                timeout=timeout,
            )

        try:
            stream = _create_stream(messages, max_tokens)
        except Exception as exc:
            err = str(exc).lower()
            if "max_tokens must be at least 1" in err or "context length" in err:
                messages, max_tokens, _ = fit_messages_to_context(
                    self.llm.model_name,
                    messages,
                    max_context_len,
                    max_tokens,
                    extra_margin=256,
                )
                stream = _create_stream(messages, max_tokens)
            else:
                raise
        for chunk in stream:
            if not hasattr(chunk, "choices"):
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", "")
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                parts.append(text)
        end = time.perf_counter()
        response = "".join(parts)

        # Compute TTFT and TPOT
        if first_token_time is None:
            self.last_ttft = 0.0
            self.last_tpot = 0.0
        else:
            ttft = first_token_time - start
            total_tokens = max(len(response.split()), 1)
            tpot = (end - start - ttft) / total_tokens
            self.last_ttft = float(ttft)
            self.last_tpot = float(tpot)

        # Track cost
        prompt_text = "".join([item["content"] for item in messages])
        cost_count(prompt_text, response, self.llm.model_name)
        return response

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """
        Run the agent.
        Args:
            inputs: dict[str, str]: Raw inputs.
            spatial_info: dict[str, dict]: Spatial information.
            temporal_info: dict[str, dict]: Temporal information.
        Returns:
            Any: str: Aggregated message.
        """
        query = input['query']
        passed, response= inner_test(input, spatial_info, temporal_info)
        if passed:
            return response
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        request_timeout = kwargs.get("request_timeout")
        default_max_out = get_model_max_output_tokens(self.llm.model_name)
        max_out = resolve_max_output_tokens(self.llm.model_name, prompt, default_max_out)
        response = self._call_llm_stream(prompt, max_tokens=max_out, request_timeout=request_timeout)
        response = post_process(input, response, self.post_process)
        logger.trace(f"Agent {self.id} Role: {self.role.role} LLM: {self.llm.model_name}")
        logger.trace(f"system prompt:\n {prompt[0]['content']}")
        logger.trace(f"user prompt:\n {prompt[1]['content']}")
        logger.trace(f"response:\n {response}")

        # #! 
        # received_id = []
        # for id, info in spatial_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')
        # for id, info in temporal_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')

        # entry = {
        #     "id": self.id,
        #     "role": self.role.role,
        #     "llm_name": self.llm.model_name,
        #     "system_prompt": prompt[0]['content'],
        #     "user_prompt": prompt[1]['content'],
        #     "received_id": received_id,
        #     "response": response,
        # }
        # try:
        #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []

        # data.append(entry)

        # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        # #!

        post_format_prompt = output_format_prompt[self.post_output_format]
        if post_format_prompt is not None:
            system_prompt = f"{self.post_description}\n"
            system_prompt += f"Format requirements that must be followed:\n{post_format_prompt}"
            user_prompt = f"{query}\nThe initial thinking information is:\n{response} \n Please refer to the new format requirements when replying."
            prompt_budget = self._prompt_budget()
            trimmed_user = self._enforce_budget(system_prompt, user_prompt, prompt_budget)
            prompt = [{'role':'system','content':system_prompt},{'role':'user','content':trimmed_user}]
            max_out = resolve_max_output_tokens(self.llm.model_name, prompt, default_max_out)
            response = self._call_llm_stream(prompt, max_tokens=max_out, request_timeout=request_timeout)
            logger.trace(f"post system prompt:\n {system_prompt}")
            logger.trace(f"post user prompt:\n {trimmed_user}")
            logger.trace(f"post response:\n {response}")
            
            # #! 
            # received_id = []
            # role = self.role.role
            # received_id.append(self.id + '(' + role + ')')

            # entry = {
            #     "id": self.id,
            #     "role": self.role.role,
            #     "llm_name": self.llm.model_name,
            #     "system_prompt": prompt[0]['content'],
            #     "user_prompt": prompt[1]['content'],
            #     "received_id": received_id,
            #     "response": response,
            # }
            # try:
            #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
            #         data = json.load(f)
            # except (FileNotFoundError, json.JSONDecodeError):
            #     data = []

            # data.append(entry)

            # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
            #     json.dump(data, f, ensure_ascii=False, indent=2)
            # #!
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None

    def _limit_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return limit_prompt_for_llm(self.llm.model_name, system_prompt, user_prompt)

@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None, agent_name = "", domain = "", llm_name = "", prompt_file = ""):
        super().__init__(id, agent_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_file = json.load(open(f"{prompt_file}", 'r', encoding='utf-8'))
        # Timing metrics for telemetry
        self.last_ttft = 0.0
        self.last_tpot = 0.0
        # Strategy injection (used only for predictor data generation)
        self.strategy_name = ""
        self.strategy_prompt = ""

    def set_strategy(self, strategy: str) -> None:
        """Set prompt strategy for predictor data generation."""
        self.strategy_name = strategy
        self.strategy_prompt = get_strategy_prompt(strategy) if strategy else ""

    def _limit_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return limit_prompt_for_llm(self.llm.model_name, system_prompt, user_prompt)

    def _call_llm_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
        request_timeout: float | None = None,
    ) -> str:
        """Call LLM with streaming to capture TTFT and TPOT metrics."""
        timeout = _normalize_request_timeout(request_timeout)
        base_client = _get_shared_sync_client(
            base_url=_resolve_base_url(self.llm.model_name),
            api_key=_resolve_api_key(),
        )
        client = base_client.with_options(timeout=timeout)
        if max_tokens is None:
            max_tokens = LLM.DEFAULT_MAX_TOKENS
        else:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = 1
            if max_tokens > LLM.DEFAULT_MAX_TOKENS:
                max_tokens = LLM.DEFAULT_MAX_TOKENS

        max_context_len = get_model_max_context_len(self.llm.model_name)
        messages, max_tokens, _ = fit_messages_to_context(
            self.llm.model_name, messages, max_context_len, max_tokens
        )

        start = time.perf_counter()
        first_token_time = None
        parts: List[str] = []
        def _create_stream(msgs, max_out):
            return client.chat.completions.create(
                model=self.llm.model_name,
                messages=msgs,
                stream=True,
                max_tokens=max_out,
                temperature=LLM.DEFAULT_TEMPERATURE,
                timeout=timeout,
            )

        try:
            stream = _create_stream(messages, max_tokens)
        except Exception as exc:
            err = str(exc).lower()
            if "max_tokens must be at least 1" in err or "context length" in err:
                messages, max_tokens, _ = fit_messages_to_context(
                    self.llm.model_name,
                    messages,
                    max_context_len,
                    max_tokens,
                    extra_margin=256,
                )
                stream = _create_stream(messages, max_tokens)
            else:
                raise
        for chunk in stream:
            if not hasattr(chunk, "choices"):
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", "")
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                parts.append(text)
        end = time.perf_counter()
        response = "".join(parts)

        # Compute TTFT and TPOT
        if first_token_time is None:
            self.last_ttft = 0.0
            self.last_tpot = 0.0
        else:
            ttft = first_token_time - start
            total_tokens = max(len(response.split()), 1)
            tpot = (end - start - ttft) / total_tokens
            self.last_ttft = float(ttft)
            self.last_tpot = float(tpot)

        # Track cost
        prompt_text = "".join([item["content"] for item in messages])
        cost_count(prompt_text, response, self.llm.model_name)
        return response

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = f"{self.prompt_file['system']}"
        if self.strategy_prompt:
            system_prompt = f"{self.strategy_prompt}\n{system_prompt}"
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        user_prompt = f"The task is:\n\n {raw_inputs['query']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str} {self.prompt_file['user']}"
        user_prompt = self._limit_prompt(system_prompt, user_prompt)
        return [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    
    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        request_timeout = kwargs.get("request_timeout")
        default_max_out = get_model_max_output_tokens(self.llm.model_name)
        max_out = resolve_max_output_tokens(self.llm.model_name, prompt, default_max_out)
        response = self._call_llm_stream(prompt, max_tokens=max_out, request_timeout=request_timeout)
        logger.trace(f"Final Refer Node LLM: {self.llm.model_name}")
        logger.trace(f"Final System Prompt:\n {prompt[0]['content']}")
        logger.trace(f"Final User Prompt:\n {prompt[1]['content']}")
        logger.trace(f"Final Response:\n {response}")
        # #! 
        # received_id = []
        # for id, info in spatial_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')
        # for id, info in temporal_info.items():
        #     role = info["role"].role
        #     received_id.append(id + '(' + role + ')')

        # entry = {
        #     "id": self.id,
        #     "role": "FinalDecision",
        #     "llm_name": self.llm.model_name,
        #     "system_prompt": prompt[0]['content'],
        #     "user_prompt": prompt[1]['content'],
        #     "received_id": received_id,
        #     "response": response,
        # }
        # try:
        #     with open(f'./result/tmp_log.json', 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []

        # data.append(entry)

        # with open(f'./result/tmp_log.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
        # #!
        return response
    
    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None
