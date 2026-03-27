"""UI utilities for Summit-Sim."""

from summit_sim.schemas import ScenarioConfig

# Standard 1-5 star rating scale for scenario evaluation
RATING_SCALE = [
    {"value": 1, "label": "⭐ (1) Poor", "description": "Unacceptable"},
    {"value": 2, "label": "⭐⭐ (2) Below Avg", "description": "Major issues"},
    {"value": 3, "label": "⭐⭐⭐ (3) Acceptable", "description": "Safe to use"},
    {"value": 4, "label": "⭐⭐⭐⭐ (4) Good", "description": "Quality scenario"},
    {"value": 5, "label": "⭐⭐⭐⭐⭐ (5) Excellent", "description": "Outstanding"},
]


def get_rating_actions() -> list[dict]:
    """Get action button specifications for the standard rating scale.

    Returns action specs that can be used with cl.AskActionMessage.
    Each action has name, payload, and label fields.
    """
    return [
        {
            "name": f"rate_{r['value']}",
            "payload": {"value": r["value"]},
            "label": r["label"],
        }
        for r in RATING_SCALE
    ]


def get_rating_content(title: str = "Rate this scenario") -> str:
    """Get the content text for a rating prompt.

    Returns a descriptive message asking the user to rate quality.
    """
    return f"#### {title}\n\nPlease evaluate the quality using the scale below:"


def get_author_form_fields() -> list[dict]:
    """Generate form field configuration from ScenarioConfig schema.

    Extracts UI metadata from json_schema_extra to create field definitions
    that can be passed directly to the frontend form component.
    """
    fields = []

    for name, field in ScenarioConfig.model_fields.items():
        if name == "class_id":
            continue  # Skip internal fields not shown in form

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
