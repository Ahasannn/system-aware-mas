from typing import Dict
import json
import os
from loguru import logger

from MAR.Agent.agent_registry import AgentRegistry
from MAR.LLM.llm_profile_test import get_model_max_context_len, get_model_max_output_tokens
from MAR.LLM.llm_registry import LLMRegistry
from MAR.LLM.price import cal_token, truncate_text_for_model
from MAR.Roles.role_registry import RoleRegistry
from MAR.Graph.node import Node
from MAR.Prompts.message_aggregation import message_aggregation,inner_test
from MAR.Prompts.post_process import post_process
from MAR.Prompts.output_format import output_format_prompt
from MAR.Prompts.reasoning import reasoning_prompt


def limit_prompt_for_llm(llm_model_name: str, system_prompt: str, user_prompt: str) -> str:
    max_model_len = get_model_max_context_len(llm_model_name)
    prompt_budget = max(128, max_model_len - 512)
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

    # Keep a small safety margin for tool / chat formatting overhead.
    available = max(0, max_model_len - prompt_tokens - 32)
    return max(16, min(int(default_max_output_tokens), available))


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

        system_prompt = f"{self.description}\n{reason_prompt}"
        if self.latency_budget:
            system_prompt += f"\nLatency budget: {self.latency_budget}. Work a bit faster and be concise."
        system_prompt += f"\nFormat requirements that must be followed:\n{format_prompt}" if format_prompt else ""
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

    def set_llm(self, llm_name: str) -> None:
        if not llm_name:
            return
        if llm_name == self.llm_name:
            return
        self.llm_name = llm_name
        self.llm = LLMRegistry.get(llm_name)

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
        response = self.llm.gen(prompt, max_tokens=max_out, request_timeout=request_timeout)
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
            response = self.llm.gen(prompt, max_tokens=max_out, request_timeout=request_timeout)
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

    def _limit_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return limit_prompt_for_llm(self.llm.model_name, system_prompt, user_prompt)

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):  
        system_prompt = f"{self.prompt_file['system']}"
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
        response = self.llm.gen(prompt, max_tokens=max_out, request_timeout=request_timeout)
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
