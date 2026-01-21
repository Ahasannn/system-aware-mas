from typing import Dict

from loguru import logger

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_registry import LLMRegistry
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

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        query = input["query"]
        passed, response = inner_test(input, spatial_info, temporal_info)
        if passed:
            return response
        prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        response = self.llm.gen(prompt)
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
            response = self.llm.gen(prompt)
            logger.trace(f"post system prompt:\n {system_prompt}")
            logger.trace(f"post user prompt:\n {user_prompt}")
            logger.trace(f"post response:\n {response}")
        return response

    def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        return None
