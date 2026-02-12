import time
from typing import Dict, List

from loguru import logger

from openai import OpenAI

from MAR.Agent.agent import limit_prompt_for_llm, resolve_max_output_tokens, fit_messages_to_context
from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_profile_full import get_model_max_context_len, get_model_max_output_tokens
from MAR.LLM.llm_registry import LLMRegistry
from MAR.LLM.llm import LLM
from MAR.LLM.gpt_chat import _resolve_api_key, _resolve_base_url
from MAR.LLM.price import cal_token, cost_count
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation import message_aggregation, inner_test
from MAR.Prompts.post_process import post_process
from MAR.Prompts.output_format import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt
from MAR.InfraMind.prompt_strategies import get_strategy_prompt


@AgentRegistry.register("InfraMindAgent")
class InfraMindAgent(Node):
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        role: str | None = None,
        llm_name: str = "",
        reason_name: str = "",
        latency_budget: str | None = None,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
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
        if reason_name == "Reflection" and self.post_output_format == "None":
            self.post_output_format = self.output_format
            self.post_description = (
                "\nReflect on possible errors in the answer above and answer again using the same format. "
                "If you think there are no errors in your previous answers that will affect the results, "
                "there is no need to correct them.\n"
            )

        self.strategy_name = ""
        self.strategy_prompt = ""
        self.prompt_base = ""
        self.last_ttft = 0.0
        self.last_tpot = 0.0
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.request_timeout = float(request_timeout) if request_timeout else None
        # EDF priority for vLLM scheduling (set by InfraMindGraph from deadline)
        self.priority: int | None = None

    def set_llm(self, llm_name: str) -> None:
        if not llm_name:
            return
        if llm_name == self.llm_name:
            return
        self.llm_name = llm_name
        self.llm = LLMRegistry.get(llm_name)

    def set_strategy(self, strategy: str) -> None:
        self.strategy_name = strategy
        self.strategy_prompt = get_strategy_prompt(strategy) if strategy else ""

    def _process_inputs(
        self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs
    ):
        query = raw_inputs["query"]
        spatial_prompt = message_aggregation(raw_inputs, spatial_info, self.message_aggregation)
        temporal_prompt = message_aggregation(raw_inputs, temporal_info, self.message_aggregation)
        format_prompt = output_format_prompt[self.output_format]
        reason_prompt = reasoning_prompt[self.reason]

        base_system = f"{self.description}\n{reason_prompt}"
        if self.latency_budget:
            base_system += f"\nLatency budget: {self.latency_budget}. Work a bit faster and be concise."
        if format_prompt:
            base_system += f"\nFormat requirements that must be followed:\n{format_prompt}"

        system_prompt = base_system
        if self.strategy_prompt:
            system_prompt = f"{self.strategy_prompt}\n{base_system}"

        user_prompt = f"{query}\n"
        if spatial_prompt:
            user_prompt += f"At the same time, other agents' outputs are as follows:\n\n{spatial_prompt}"
        if temporal_prompt:
            user_prompt += f"\n\nIn the last round of dialogue, other agents' outputs were:\n\n{temporal_prompt}"

        trimmed_user_prompt = limit_prompt_for_llm(self.llm.model_name, system_prompt, user_prompt)
        # Only log if significant trimming occurred (>20% reduction)
        if trimmed_user_prompt != user_prompt:
            orig_tokens = cal_token(self.llm.model_name, user_prompt)
            new_tokens = cal_token(self.llm.model_name, trimmed_user_prompt)
            if orig_tokens > 0 and (orig_tokens - new_tokens) / orig_tokens > 0.2:
                logger.warning(
                    "Significant prompt trimming for {}: {} → {} tokens ({:.1f}% reduction)",
                    self.llm.model_name.split("/")[-1],
                    orig_tokens,
                    new_tokens,
                    (orig_tokens - new_tokens) / orig_tokens * 100,
                )
        user_prompt = trimmed_user_prompt

        self.prompt_base = f"{base_system}\n\n{user_prompt}".strip()
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _call_llm_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int | None = None,
        request_timeout: float | None = None,
    ) -> str:
        timeout = request_timeout if request_timeout is not None else self.request_timeout
        client = OpenAI(
            base_url=_resolve_base_url(self.llm_name),
            api_key=_resolve_api_key(),
            timeout=timeout,
        )
        prompt_tokens = sum(
            cal_token(self.llm_name, msg.get("content", "")) for msg in messages if isinstance(msg, dict)
        )
        max_context_len = get_model_max_context_len(self.llm_name)
        default_max_out = self.max_tokens or get_model_max_output_tokens(self.llm_name)
        allowed_max = resolve_max_output_tokens(self.llm_name, messages, default_max_out)
        requested_max = None
        if max_tokens is not None:
            requested_max = int(max_tokens)
            if requested_max <= 0:
                requested_max = 1
            # Silently limit to requested max_tokens if specified
            allowed_max = min(requested_max, allowed_max)

        # Safety margin for token counting mismatches vs the server tokenizer (e.g., vLLM).
        safe_available = max(0, max_context_len - prompt_tokens - 32)
        if safe_available <= 0:
            logger.warning(
                "Prompt for %s already consumes %d/%d tokens; forcing max_tokens=1",
                self.llm_name,
                prompt_tokens,
                max_context_len,
            )
            allowed_max = 1
        elif allowed_max > safe_available:
            # Silently adjust max_tokens to fit within context window
            allowed_max = safe_available
        max_tokens = max(1, allowed_max)
        messages, max_tokens, prompt_tokens = fit_messages_to_context(
            self.llm_name, messages, max_context_len, max_tokens
        )
        start = time.perf_counter()
        first_token_time = None
        parts: List[str] = []
        extra_body = {"priority": self.priority} if self.priority is not None else None
        def _create_stream(msgs, max_out):
            return client.chat.completions.create(
                model=self.llm_name,
                messages=msgs,
                stream=True,
                max_tokens=max_out,
                temperature=LLM.DEFAULT_TEMPERATURE,
                timeout=timeout,
                extra_body=extra_body,
            )

        try:
            stream = _create_stream(messages, max_tokens)
        except Exception as exc:
            err = str(exc).lower()
            if "max_tokens must be at least 1" in err or "context length" in err:
                messages, max_tokens, _ = fit_messages_to_context(
                    self.llm_name,
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
        if first_token_time is None:
            self.last_ttft = 0.0
            self.last_tpot = 0.0
        else:
            ttft = first_token_time - start
            total_tokens = max(len(response.split()), 1)
            tpot = (end - start - ttft) / total_tokens
            self.last_ttft = float(ttft)
            self.last_tpot = float(tpot)
        prompt_text = "".join([item["content"] for item in messages])
        cost_count(prompt_text, response, self.llm_name)
        return response

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        query = input["query"]
        passed, response = inner_test(input, spatial_info, temporal_info)
        if passed:
            return response
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        max_tokens = kwargs.get("max_tokens")
        request_timeout = kwargs.get("request_timeout")
        response = self._call_llm_stream(prompt, max_tokens=max_tokens, request_timeout=request_timeout)
        response = post_process(input, response, self.post_process)
        logger.trace(f"Agent {self.id} Role: {self.role.role} LLM: {self.llm.model_name}")
        logger.trace(f"system prompt:\n {prompt[0]['content']}")
        logger.trace(f"user prompt:\n {prompt[1]['content']}")
        logger.trace(f"response:\n {response}")

        post_format_prompt = output_format_prompt[self.post_output_format]
        if post_format_prompt is not None:
            system_prompt = f"{self.post_description}\n"
            if self.strategy_prompt:
                system_prompt = f"{self.strategy_prompt}\n{system_prompt}"
            system_prompt += f"Format requirements that must be followed:\n{post_format_prompt}"
            user_prompt = (
                f"{query}\nThe initial thinking information is:\n{response} \n "
                "Please refer to the new format requirements when replying."
            )
            user_prompt = limit_prompt_for_llm(self.llm.model_name, system_prompt, user_prompt)
            prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response = self._call_llm_stream(prompt, max_tokens=max_tokens, request_timeout=request_timeout)
            logger.trace(f"post system prompt:\n {system_prompt}")
            logger.trace(f"post user prompt:\n {user_prompt}")
            logger.trace(f"post response:\n {response}")
        return response

    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None
