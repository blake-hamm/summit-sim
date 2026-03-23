"""Shared agent configuration and provider setup."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterReasoning,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

from summit_sim.settings import settings


@lru_cache(maxsize=1)
def get_provider() -> OpenRouterProvider:
    """Get singleton OpenRouter provider instance.

    Uses lazy initialization to avoid errors during module import
    when OPENROUTER_API_KEY is not set (e.g., during testing).

    Returns:
        Shared OpenRouterProvider instance

    """
    return OpenRouterProvider(api_key=settings.openrouter_api_key)


# Module-level agent container for lazy initialization
_agent_container: dict[str, Any] = {}


def get_agent(
    agent_name: str,
    output_type: type,
    system_prompt: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> Agent[Any, Any]:
    """Get or create an agent singleton by name.

    Args:
        agent_name: Unique name for the agent (used for caching)
        output_type: Pydantic model for structured output
        system_prompt: System prompt for the agent
        reasoning_effort: Reasoning level ("low", "medium", "high")

    Returns:
        Configured Agent instance

    """
    if agent_name not in _agent_container:
        reasoning_config: OpenRouterReasoning = {"effort": reasoning_effort}  # type: ignore[typeddict-item]
        _agent_container[agent_name] = Agent(
            OpenRouterModel(
                settings.default_model,
                provider=get_provider(),
            ),
            output_type=output_type,
            system_prompt=system_prompt,
            model_settings=OpenRouterModelSettings(
                openrouter_reasoning=reasoning_config
            ),
        )
    return _agent_container[agent_name]
