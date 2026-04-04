"""Shared agent configuration and provider setup."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, cast

import mlflow
from google.genai import types
from google.oauth2 import service_account
from mlflow.entities.model_registry.prompt_version import PromptVersion
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from summit_sim.settings import settings

logger = logging.getLogger(__name__)

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

ALL_WFR_OBJECTIVES = [
    obj for category in WFR_LEARNING_OBJECTIVES.values() for obj in category
]

REASONING_TO_THINKING: dict[Literal["low", "medium", "high"], types.ThinkingLevel] = {
    "low": types.ThinkingLevel.MINIMAL,
    "medium": types.ThinkingLevel.LOW,
    "high": types.ThinkingLevel.HIGH,
}


@lru_cache(maxsize=1)
def get_provider() -> GoogleProvider:
    """Get singleton Google Provider instance for Vertex AI."""
    credentials = service_account.Credentials.from_service_account_file(
        str(settings.google_application_credentials),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return GoogleProvider(
        credentials=credentials,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )


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
        thinking_level = REASONING_TO_THINKING.get(
            reasoning_effort, types.ThinkingLevel.LOW
        )
        model_settings = GoogleModelSettings(
            google_thinking_config=types.ThinkingConfigDict(
                thinking_level=thinking_level
            )
        )
        agent = Agent(
            GoogleModel(settings.default_model, provider=get_provider()),
            output_type=output_type,
            system_prompt=cast(str, system_prompt_obj.template),
            model_settings=model_settings,
        )
        _agent_container[agent_name] = (agent, user_prompt_obj)

    agent, user_prompt = _agent_container[agent_name]
    return agent, user_prompt
