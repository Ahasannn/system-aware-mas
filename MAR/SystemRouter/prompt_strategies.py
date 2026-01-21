from enum import Enum
from typing import Dict, List, Optional


class PromptStrategy(str, Enum):
    FLASH = "Flash"
    CONCISE = "Concise"
    DEEP_THINK = "DeepThink"


_TEMPLATES: Dict[PromptStrategy, str] = {
    PromptStrategy.FLASH: "Answer quickly with the minimal steps needed. Avoid extra prose. Respond in one or two sentences.",
    PromptStrategy.CONCISE: "Provide a short but careful answer. Show the key reasoning in 2-3 bullet points before the final answer.",
    PromptStrategy.DEEP_THINK: "Think step by step with a brief plan, then provide the final answer with justification. Be explicit about assumptions.",
}

def get_strategy_prompt(strategy: str) -> str:
    try:
        strategy_enum = PromptStrategy(strategy)
    except ValueError:
        strategy_enum = PromptStrategy.CONCISE
    return _TEMPLATES.get(strategy_enum, _TEMPLATES[PromptStrategy.CONCISE])


def build_messages(query: str, strategy: str, role: Optional[str] = None, context: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Produce OpenAI-style messages with a strategy-specific system prompt.
    """
    try:
        strategy_enum = PromptStrategy(strategy)
    except ValueError:
        strategy_enum = PromptStrategy.CONCISE

    system_prompt = _TEMPLATES.get(strategy_enum, _TEMPLATES[PromptStrategy.CONCISE])
    if role:
        system_prompt = f"{system_prompt}\n\nRole: {role}"

    user_content = query
    if context:
        user_content = f"{query}\n\nContext from previous nodes:\n{context}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
