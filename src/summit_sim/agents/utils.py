"""Shared agent configuration and provider setup."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, cast

import mlflow
from mlflow.entities.model_registry.prompt_version import PromptVersion
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
    """Get singleton OpenRouter provider instance."""
    return OpenRouterProvider(api_key=settings.openrouter_api_key)


# Module-level agent container for lazy initialization
_agent_container: dict[str, Any] = {}


def _get_or_register_prompt(prompt_name: str, template: str) -> PromptVersion:
    """Get prompt from registry, registering new version if changed."""
    prompt_uri = f"prompts:/{prompt_name}@latest"

    try:
        loaded = mlflow.genai.load_prompt(prompt_uri)  # type: ignore[attr-defined]
        if loaded.template != template:
            logger.info("Prompt changed for %s, registering new version", prompt_name)
            mlflow.genai.register_prompt(  # type: ignore[attr-defined]
                name=prompt_name,
                template=template,
            )
    except Exception:
        logger.info("Registering new prompt for %s", prompt_name)
        mlflow.genai.register_prompt(  # type: ignore[attr-defined]
            name=prompt_name,
            template=template,
        )

    return mlflow.genai.load_prompt(prompt_uri)  # type: ignore[attr-defined]


def setup_agent_and_prompts(
    agent_name: str,
    output_type: type,
    system_prompt: str,
    user_prompt_template: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> tuple[Agent[Any, Any], PromptVersion]:
    """Create/configure agent with versioned prompts."""
    if agent_name not in _agent_container:
        system_prompt_obj = _get_or_register_prompt(
            f"{agent_name}-system", system_prompt
        )
        _agent_container[agent_name] = Agent(
            OpenRouterModel(
                settings.default_model,
                provider=get_provider(),
            ),
            output_type=output_type,
            system_prompt=cast(str, system_prompt_obj.template),
            model_settings=OpenRouterModelSettings(
                openrouter_reasoning={"effort": reasoning_effort},
                openrouter_usage={"include": True},
            ),
        )

    agent = _agent_container[agent_name]
    user_prompt = _get_or_register_prompt(f"{agent_name}-user", user_prompt_template)
    return agent, user_prompt
