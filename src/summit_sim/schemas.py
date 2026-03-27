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


class ChoiceOption(BaseModel):
    """A multiple choice option for a scenario turn.

    Each option includes the action description and the next turn ID
    that follows if this option is selected.
    """

    choice_id: str = Field(..., description="Unique identifier for this choice")
    description: str = Field(..., description="Description of the action")
    is_correct: bool = Field(
        ..., description="Whether this is the medically optimal choice"
    )
    next_turn_id: int | None = Field(
        default=None, description="ID of next turn (null if scenario ends)"
    )


class ScenarioTurn(BaseModel):
    """A single turn in a pre-generated scenario.

    Contains the narrative setup and multiple choice options for the student.
    The actual simulation uses these pre-written turns, but AI generates
    personalized feedback when a choice is made.
    """

    turn_id: int = Field(..., description="Unique identifier for this turn")
    narrative_text: str = Field(
        ..., description="Story description of the current situation"
    )
    hidden_state: dict[str, str] = Field(
        default_factory=dict,
        description="Secret medical information not visible to students",
    )
    scene_state: dict[str, str] = Field(
        default_factory=dict, description="Current visible scene conditions"
    )
    choices: list[ChoiceOption] = Field(
        ...,
        description="3-5 multiple choice actions available",
        min_length=3,
        max_length=5,
    )


class ScenarioDraft(BaseModel):
    """Complete AI-generated wilderness rescue scenario.

    Generated from TeacherConfig and contains all turns pre-written.
    The scenario follows a branching path based on student choices,
    but all content is generated upfront.
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
    turns: list[ScenarioTurn] = Field(
        ..., description="All turns in this scenario (at least 3)", min_length=3
    )

    def get_turn(self, turn_id: int) -> ScenarioTurn | None:
        """Get a turn by its ID."""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def get_starting_turn(self) -> ScenarioTurn | None:
        """Get the starting turn (turn_id=0)."""
        return self.get_turn(0)


class SimulationResult(BaseModel):
    """Result after a student makes a choice in the simulation.

    Contains AI-generated personalized feedback and the next turn
    (or completion status).
    """

    selected_choice: ChoiceOption = Field(
        ..., description="The choice the student selected"
    )
    feedback: str = Field(
        ..., description="AI-generated personalized feedback on the choice"
    )
    learning_moments: list[str] = Field(
        ..., description="Educational insights from this turn"
    )
    next_turn: ScenarioTurn | None = Field(
        default=None, description="Next turn (null if scenario complete)"
    )
    is_complete: bool = Field(
        ..., description="Whether the scenario has reached a conclusion"
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
