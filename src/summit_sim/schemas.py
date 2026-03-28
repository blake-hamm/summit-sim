"""Pydantic schemas for Summit-Sim data models."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


def generate_scenario_id() -> str:
    """Generate unique scenario identifier."""
    return f"scn-{uuid.uuid4().hex[:8]}"


class ScenarioConfig(BaseModel):
    """Configuration provided by the author to generate a targeted WFR scenario.

    Replaces generic inputs with dimensions that directly impact WFR decision-making:
    patient assessment (Medical/Trauma/Environmental), resource management (group size),
    and evacuation logistics (distance/environment).
    """

    primary_focus: Literal["Trauma", "Medical", "Environmental", "Mixed"] = Field(
        ...,
        description="The core WFR syllabus category to test.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Curriculum Focus",
                "options": ["Trauma", "Medical", "Environmental", "Mixed"],
                "value": "Trauma",
            }
        },
    )
    environment: Literal[
        "Alpine/Mountain", "Desert", "Forest/Trail", "Water/River", "Winter/Snow"
    ] = Field(
        ...,
        description="The physical setting, driving MOI and weather risks.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Environment",
                "options": [
                    "Alpine/Mountain",
                    "Desert",
                    "Forest/Trail",
                    "Water/River",
                    "Winter/Snow",
                ],
                "value": "Alpine/Mountain",
            }
        },
    )
    available_personnel: Literal[
        "Solo Rescuer (1)", "Partner (2)", "Small Group (3-5)", "Large Expedition (6+)"
    ] = Field(
        ...,
        description="Total conscious people. Dictates litter carries or runners.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Group Size / Resources",
                "options": [
                    "Solo Rescuer (1)",
                    "Partner (2)",
                    "Small Group (3-5)",
                    "Large Expedition (6+)",
                ],
                "value": "Small Group (3-5)",
            }
        },
    )
    evac_distance: Literal[
        "Short (< 2 hours)", "Remote (1 day)", "Expedition (2+ days)"
    ] = Field(
        ...,
        description="Distance to definitive care. Key for stay vs go decisions.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Evacuation Distance",
                "options": [
                    "Short (< 2 hours)",
                    "Remote (1 day)",
                    "Expedition (2+ days)",
                ],
                "value": "Remote (1 day)",
            }
        },
    )
    complexity: Literal["Standard", "Complicated", "Critical"] = Field(
        ...,
        description="Complicated=underlying condition; Critical=deteriorating vitals.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Patient Complexity",
                "options": ["Standard", "Complicated", "Critical"],
                "value": "Standard",
            }
        },
    )


class DynamicTurnResult(BaseModel):
    """Result from ActionResponder agent after evaluating student action.

    Single schema enforces evaluation → narrative → state evolution order.
    Generated dynamically for each student action in free-text simulation.
    """

    was_correct: bool = Field(
        ..., description="Whether the student's action was medically correct"
    )
    completion_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Progress toward scenario completion (0.0-1.0 scale)",
    )
    is_complete: bool = Field(
        ..., description="Whether the scenario has reached a natural conclusion"
    )
    feedback: str = Field(
        ..., description="AI-generated personalized feedback on the action"
    )
    narrative_text: str = Field(
        ..., description="Generated narrative describing the outcome"
    )
    updated_hidden_state: str = Field(
        ...,
        description="Updated secret medical info after this turn (full replacement)",
    )
    updated_scene_state: str = Field(
        ...,
        description="Updated scene conditions after this turn (full replacement)",
    )


class ScenarioDraft(BaseModel):
    """Complete AI-generated wilderness rescue scenario.

    Generated from ScenarioConfig for dynamic open-ended simulation.
    Contains only initial setup - turns are generated dynamically
    based on student free-text actions.
    """

    title: str = Field(..., description="Short name for the scenario")
    setting: str = Field(..., description="Location and environmental context")
    patient_summary: str = Field(..., description="Brief description of the patient")
    hidden_truth: str = Field(
        ..., description="Secret medical information not visible to students"
    )
    learning_objectives: list[str] = Field(
        ..., description="List of skills students should practice"
    )
    initial_narrative: str = Field(
        ..., description="Opening narrative that sets the scene for the student"
    )
    hidden_state: str = Field(
        ...,
        description="Initial secret medical info for AI reference",
    )
    scene_state: str = Field(
        ...,
        description="Initial visible scene conditions",
    )


class DebriefReport(BaseModel):
    """Structured debrief report analyzing student simulation performance.

    Generated after simulation completion to provide comprehensive feedback
    on student decision-making, learning opportunities, and performance metrics.
    """

    summary: str = Field(..., description="Executive summary of the simulation run")
    key_mistakes: list[str] = Field(
        ..., description="Critical errors made during the simulation"
    )
    strong_actions: list[str] = Field(
        ..., description="Decisions the student handled well"
    )
    best_next_actions: list[str] = Field(
        ..., description="Recommendations for future scenarios"
    )
    teaching_points: list[str] = Field(
        ..., description="Key learning concepts to reinforce"
    )
    completion_status: Literal["pass", "fail"] = Field(
        ..., description="Overall pass/fail based on performance"
    )
    final_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage score (correct choices / total turns * 100)",
    )
