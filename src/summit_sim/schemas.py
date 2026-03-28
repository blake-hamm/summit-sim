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

    mode: Literal["instructor", "student"] = Field(
        default="instructor",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Role",
                "options": ["Instructor (Review & Share)", "Student (Play Now)"],
                "value": "Instructor (Review & Share)",
            }
        },
    )
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

    title: str = Field(
        ...,
        description=(
            "A punchy, unique title for the scenario. "
            "Examples: 'Lightning Strike on the Grand Teton', "
            "'Anaphylaxis on the Pine Ridge Trail', 'HAPE at 14,000 Feet'"
        ),
    )
    setting: str = Field(
        ...,
        description=(
            "Specific location, weather, and time of day. Keep under 20 words. "
            "Example: 'Exposed rocky ridge at 13,000 ft. Incoming thunderstorm, "
            "dropping temperatures.'"
        ),
    )
    patient_summary: str = Field(
        ...,
        description=(
            "Age, sex, chief complaint, and visible mechanism of injury (MOI). "
            "Example: '28-year-old female, thrown 10 feet by indirect lightning "
            "strike. Conscious but confused.'"
        ),
    )
    hidden_truth: str = Field(
        ...,
        description=(
            "The actual medical diagnosis students must discover. Example: 'Patient "
            "has a minor burn on the right leg, but the critical hidden issue is a "
            "suspected cervical spine injury from the throw and developing "
            "hypothermia.'"
        ),
    )
    learning_objectives: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description=(
            "2-3 specific WFR skills tested from the curriculum (e.g., 'Spinal "
            "clearance protocol', 'Lightning strike safety/evacuation', "
            "'Hypothermia prevention'). Select from WFR learning objectives catalog."
        ),
    )
    initial_narrative: str = Field(
        ...,
        max_length=400,
        description=(
            "Immersive opening scene ending with a call to action. STRICTLY 2-4 "
            "sentences. DO NOT EXCEED 4 SENTENCES. Present tense, second person "
            "('You are...'). Must end with a question inviting action. Example: 'You "
            "are descending the Grand Teton when a loud crack echoes, and you see "
            "your climbing partner thrown against the rocks by an indirect lightning "
            "strike. The sky is dark, and the wind is picking up. She is groaning on "
            "the ground. What is your first move?'"
        ),
    )
    hidden_state: str = Field(
        ...,
        description=(
            "Comprehensive baseline medical data for AI reference. Must include "
            "complete initial vitals (HR, RR, BP, SCTM, AVPU), SAMPLE history, and "
            "hidden injuries. Write as a clinical summary. Example: 'Patient is "
            "A&O x2. HR 110, RR 24, BP 130/80. SCTM: Pale, cool, clammy. "
            "Superficial fern-like burn on right calf. Tenderness upon palpation "
            "of C4 vertebrae. No other major trauma. Cannot recall the incident.'"
        ),
    )
    scene_state: str = Field(
        ...,
        description=(
            "Environmental context, available gear, group dynamics, and exact "
            "distance/time to definitive care. Keep concise. Example: 'Group of 2. "
            "1 rope, standard rack, basic WFR first aid kit. 6 hours from the "
            "trailhead. Immediate danger of secondary lightning strikes.'"
        ),
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
