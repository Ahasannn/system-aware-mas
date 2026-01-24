import time
from typing import Dict, List

from loguru import logger

from openai import OpenAI

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_registry import LLMRegistry
from MAR.LLM.llm import LLM
from MAR.LLM.gpt_chat import _resolve_api_key, _resolve_base_url
from MAR.LLM.price import cost_count
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation import message_aggregation, inner_test
from MAR.Prompts.post_process import post_process
from MAR.Prompts.output_format import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt
from MAR.SystemRouter.prompt_strategies import get_strategy_prompt


@AgentRegistry.register("SystemRouterAgent")
class SystemRouterAgent(Node):
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
        if max_tokens is None:
            max_tokens = self.max_tokens or LLM.DEFAULT_MAX_TOKENS
        else:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = 1
            if max_tokens > LLM.DEFAULT_MAX_TOKENS:
                max_tokens = LLM.DEFAULT_MAX_TOKENS
        start = time.perf_counter()
        first_token_time = None
        parts: List[str] = []
        stream = client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=LLM.DEFAULT_TEMPERATURE,
            timeout=timeout,
        )
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
            prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response = self._call_llm_stream(prompt, max_tokens=max_tokens, request_timeout=request_timeout)
            logger.trace(f"post system prompt:\n {system_prompt}")
            logger.trace(f"post user prompt:\n {user_prompt}")
            logger.trace(f"post response:\n {response}")
        return response

    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None
