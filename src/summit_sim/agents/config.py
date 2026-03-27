"""Shared agent configuration and provider setup."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

import mlflow
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

from summit_sim.settings import settings

logger = logging.getLogger(__name__)


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


def _get_or_register_system_prompt(agent_name: str, system_prompt: str) -> str:
    """Get system prompt from registry, registering if needed.

    Args:
        agent_name: Unique name for the agent
        system_prompt: System prompt content to register if not exists

    Returns:
        System prompt template from registry

    """
    prompt_name = f"{agent_name}-system"
    prompt_uri = f"prompts:/{prompt_name}@latest"

    try:
        # Try to load existing prompt
        loaded = mlflow.genai.load_prompt(prompt_uri)  # type: ignore[attr-defined]
        return str(loaded.template)
    except Exception:
        # Prompt doesn't exist, register it
        mlflow.genai.register_prompt(  # type: ignore[attr-defined]
            name=prompt_name,
            template=system_prompt,
        )
        # Return the original since we just registered it
        return system_prompt


def get_agent(
    agent_name: str,
    output_type: type,
    system_prompt: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> Agent[Any, Any]:
    """Get or create an agent singleton by name.

    System prompts are loaded from MLflow Prompt Registry.
    Prompts are auto-registered on first use if they don't exist.

    Args:
        agent_name: Unique name for the agent (used for caching and prompt lookup)
        output_type: Pydantic model for structured output
        system_prompt: System prompt content (registered if not exists)
        reasoning_effort: Reasoning level ("low", "medium", "high")

    Returns:
        Configured Agent instance

    """
    if agent_name not in _agent_container:
        loaded_prompt = _get_or_register_system_prompt(agent_name, system_prompt)
        _agent_container[agent_name] = Agent(
            OpenRouterModel(
                settings.default_model,
                provider=get_provider(),
            ),
            output_type=output_type,
            system_prompt=loaded_prompt,
            model_settings=OpenRouterModelSettings(
                openrouter_reasoning={"effort": reasoning_effort},
                openrouter_usage={"include": True},
            ),
        )
    return _agent_container[agent_name]
