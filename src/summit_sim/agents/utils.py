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

# WFR Curriculum Learning Objectives - organized by category
WFR_LEARNING_OBJECTIVES = {
    "patient_assessment": [
        "Execute systematic scene size-up identifying mechanism of injury, "
        "number of patients, and immediate hazards before approaching",
        "Perform primary assessment using ABCDE protocol identifying and "
        "correcting immediate life threats within 90 seconds",
        "Distinguish between patent and compromised airway recognizing signs "
        "of obstruction and positioning appropriately",
        "Assess breathing effectiveness counting respiratory rate and "
        "identifying inadequate ventilation requiring intervention",
        "Evaluate circulation status checking pulses, skin color/temperature, "
        "and capillary refill to identify shock",
        "Conduct focused secondary assessment performing thorough head-to-toe "
        "or focused exam based on chief complaint",
        "Obtain accurate vital signs measuring heart rate, respiratory rate, "
        "blood pressure, and temperature correctly",
    ],
    "trauma": [
        "Perform spinal motion restriction decision-making using focused spine "
        "assessment criteria to determine need for immobilization",
        "Control life-threatening hemorrhage applying direct pressure, "
        "pressure bandages, and tourniquets appropriately",
        "Manage penetrating chest wounds recognizing tension pneumothorax "
        "signs and applying occlusive dressings",
        "Assess and splint long bone fractures using improvised materials to "
        "immobilize distal joints above and below injury",
        "Evaluate traumatic brain injuries identifying concussion signs, "
        "worsening mental status, and evacuation criteria",
        "Perform wound cleaning and closure irrigating wounds and determining "
        "appropriate closure methods for wilderness context",
        "Assess and manage burns calculating TBSA, determining depth, and "
        "providing appropriate cooling and dressing",
    ],
    "environmental": [
        "Recognize and treat hypothermia identifying stages (mild/moderate/"
        "severe) and applying appropriate rewarming techniques",
        "Identify frostbite and implement rewarming distinguishing between "
        "superficial and deep frostbite and rewarming only when no refreezing "
        "risk exists",
        "Assess altitude illness progression differentiating between AMS, "
        "HACE, and HAPE with appropriate descent decisions",
        "Manage heat-related illnesses distinguishing heat exhaustion from "
        "heat stroke and initiating rapid cooling",
        "Respond to lightning strike injuries triaging multiple victims and "
        "managing cardiac/respiratory arrest patterns",
        "Treat envenomation injuries identifying venomous snake/insect bites "
        "and applying appropriate first aid while preventing further harm",
    ],
    "medical": [
        "Recognize anaphylaxis identifying multi-system allergic reactions "
        "and administering epinephrine auto-injectors",
        "Manage severe asthma attacks assessing respiratory distress severity "
        "and positioning/medicating appropriately",
        "Assess diabetic emergencies distinguishing hypoglycemia from "
        "hyperglycemia and providing appropriate glucose intervention",
        "Identify acute abdominal emergencies recognizing signs of "
        "appendicitis, bowel obstruction, or internal bleeding requiring "
        "evacuation",
        "Evaluate cardiac chest pain assessing for MI symptoms and managing "
        "patient while planning evacuation",
    ],
    "evacuation": [
        "Apply stay-vs-go decision criteria weighing patient condition, "
        "resources, environment, and evacuation logistics",
        "Package patient for litter carry securing patient to litter while "
        "protecting injuries and preparing for rough terrain",
        "Coordinate group resources delegating tasks effectively based on "
        "group size and skill levels",
        "Document patient care maintaining accurate SOAP notes and vital sign "
        "trends for handoff to definitive care",
        "Communicate evacuation needs clearly relaying accurate patient "
        "status, location, and resource requirements to rescue services",
    ],
}

# Flattened list for easy validation
ALL_WFR_OBJECTIVES = [
    obj for category in WFR_LEARNING_OBJECTIVES.values() for obj in category
]


@lru_cache(maxsize=1)
def get_provider() -> OpenRouterProvider:
    """Get singleton OpenRouter provider instance."""
    return OpenRouterProvider(api_key=settings.openrouter_api_key)


# Module-level agent container for lazy initialization
# Stores tuple of (agent, user_prompt) to avoid repeated MLflow calls
_agent_container: dict[str, tuple[Agent[Any, Any], PromptVersion]] = {}


def _get_or_register_prompt(
    prompt_name: str, template: str, register: bool = True
) -> PromptVersion:
    """Get prompt from registry, optionally registering new version if changed."""
    prompt_uri = f"prompts:/{prompt_name}@latest"

    if not register:
        return mlflow.genai.load_prompt(prompt_uri)  # type: ignore[attr-defined]

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


def setup_agent_and_prompts(  # noqa: PLR0913
    agent_name: str,
    output_type: type,
    system_prompt: str,
    user_prompt_template: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    register: bool = True,
) -> tuple[Agent[Any, Any], PromptVersion]:
    """Create/configure agent with versioned prompts."""
    if agent_name not in _agent_container:
        system_prompt_obj = _get_or_register_prompt(
            f"{agent_name}-system", system_prompt, register=register
        )
        user_prompt_obj = _get_or_register_prompt(
            f"{agent_name}-user", user_prompt_template, register=register
        )
        agent = Agent(
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
        # Cache both the agent and the prompt to eliminate network overhead
        _agent_container[agent_name] = (agent, user_prompt_obj)

    agent, user_prompt = _agent_container[agent_name]
    return agent, user_prompt


def initialize_agents() -> None:
    """Eagerly initialize all agents at server startup.

    This eliminates cold-start latency and race conditions from lazy initialization.
    Called once from on_app_startup().

    Imports are local to avoid circular dependencies at module load time.
    """
    # ruff: noqa: I001, PLC0415
    from summit_sim.agents.action_responder import (
        AGENT_NAME as ACTION_AGENT_NAME,
        SYSTEM_PROMPT as ACTION_SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE as ACTION_USER_PROMPT,
        ActionResponse,
    )
    from summit_sim.agents.debrief import (
        AGENT_NAME as DEBRIEF_AGENT_NAME,
        SYSTEM_PROMPT as DEBRIEF_SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE as DEBRIEF_USER_PROMPT,
    )
    from summit_sim.agents.generator import (
        AGENT_NAME as GENERATOR_AGENT_NAME,
        SYSTEM_PROMPT as GENERATOR_SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE as GENERATOR_USER_PROMPT,
    )
    from summit_sim.schemas import DebriefReport, ScenarioDraft

    logger.info("Initializing agents...")

    # Generator agent - high reasoning for scenario creation
    setup_agent_and_prompts(
        agent_name=GENERATOR_AGENT_NAME,
        output_type=ScenarioDraft,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt_template=GENERATOR_USER_PROMPT,
        reasoning_effort="high",
    )
    logger.debug("Generator agent initialized")

    # Debrief agent - medium reasoning for post-simulation analysis
    setup_agent_and_prompts(
        agent_name=DEBRIEF_AGENT_NAME,
        output_type=DebriefReport,
        system_prompt=DEBRIEF_SYSTEM_PROMPT,
        user_prompt_template=DEBRIEF_USER_PROMPT,
        reasoning_effort="medium",
    )
    logger.debug("Debrief agent initialized")

    # Action responder - default reasoning, no MLflow registration (uses register=False)
    setup_agent_and_prompts(
        agent_name=ACTION_AGENT_NAME,
        output_type=ActionResponse,
        system_prompt=ACTION_SYSTEM_PROMPT,
        user_prompt_template=ACTION_USER_PROMPT,
        register=False,
    )
    logger.debug("Action responder agent initialized")

    logger.info("All agents initialized successfully")
