"""UI utilities for Summit-Sim."""

from typing import Any

from pydantic import BaseModel

from summit_sim.schemas import ScenarioConfig, ScenarioDraft


def get_config_defaults(model: type[BaseModel]) -> dict[str, Any]:
    """Extract default values from schema's json_schema_extra UI metadata."""
    defaults = {}
    for name, field in model.model_fields.items():
        ui = field.json_schema_extra.get("ui", {}) if field.json_schema_extra else {}
        defaults[name] = ui.get("value")
    return defaults


def format_scenario_intro(scenario: ScenarioDraft) -> str:
    """Format the scenario intro content (excluding narrative).

    Creates a standardized display of scenario information for both
    instructor and student modes. Opening narrative is intentionally
    excluded and shown separately during simulation.
    """
    objectives_text = "\n".join(f"• {obj}" for obj in scenario.learning_objectives)
    scene_display = (
        scenario.scene_state if scenario.scene_state else "*No special conditions*"
    )

    return f"""## 🏔️ {scenario.title}

#### 🎯 Learning Objectives
{objectives_text}

#### 🏔️ Environment
**Setting:** {scenario.setting}

**Scene State:** {scene_display}

#### 🏥 Patient
**Summary:** {scenario.patient_summary}"""


SCENARIO_CONFIG_DEFAULTS: dict[str, Any] = get_config_defaults(ScenarioConfig)

# Standard 1-5 star rating scale for scenario evaluation
RATING_SCALE = [
    {"value": 1, "label": "⭐ (1) Poor", "description": "Unacceptable"},
    {"value": 2, "label": "⭐⭐ (2) Below Avg", "description": "Major issues"},
    {"value": 3, "label": "⭐⭐⭐ (3) Acceptable", "description": "Safe to use"},
    {"value": 4, "label": "⭐⭐⭐⭐ (4) Good", "description": "Quality scenario"},
    {"value": 5, "label": "⭐⭐⭐⭐⭐ (5) Excellent", "description": "Outstanding"},
]


def get_review_actions() -> list[dict]:
    """Get action button specifications for approve/revise workflow.

    Returns action specs for the AI Co-Author experience.
    """
    return [
        {
            "name": "approve",
            "payload": {"action": "approve"},
            "label": "✅ Approve & Publish",
        },
        {
            "name": "revise",
            "payload": {"action": "revise"},
            "label": "🔄 Revise Scenario",
        },
    ]


def get_review_content() -> str:
    """Get the content text for the review prompt.

    Returns a message asking the author to approve or revise the scenario.
    """
    return "#### Review Your Scenario\n\nDoes this scenario meet your needs?"


def get_satisfaction_actions() -> list[dict]:
    """Get action button specifications for post-approval satisfaction rating.

    Returns action specs that can be used with cl.AskActionMessage.
    Each action has name, payload, and label fields.
    """
    return [
        {
            "name": f"satisfaction_{r['value']}",
            "payload": {"value": r["value"]},
            "label": r["label"],
        }
        for r in RATING_SCALE
    ]


def get_satisfaction_content() -> str:
    """Get the content text for satisfaction rating prompt.

    Returns a descriptive message asking the user to rate satisfaction.
    """
    return (
        "#### How satisfied are you with the final scenario?\n\nRate your experience:"
    )


def get_author_form_fields() -> list[dict]:
    """Generate form field configuration from ScenarioConfig schema.

    Extracts UI metadata from json_schema_extra to create field definitions
    that can be passed directly to the frontend form component.
    """
    fields = []

    for name, field in ScenarioConfig.model_fields.items():
        ui = (
            field.json_schema_extra.get("ui", {})  # type: ignore[union-attr]
            if field.json_schema_extra
            else {}
        )

        fields.append(
            {
                "id": name,
                "type": ui.get("type", "text"),
                "label": ui.get("label", name.replace("_", " ").title()),
                "options": ui.get("options", []),
                "required": field.is_required(),
            }
        )

    return fields
