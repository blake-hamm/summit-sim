"""Pydantic schemas for Summit-Sim data models."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


def generate_scenario_id() -> str:
    """Generate unique scenario identifier.

    Returns:
        Scenario ID in format 'scn-{8-char hex}' (e.g., 'scn-a3f8d2e9').

    """
    return f"scn-{uuid.uuid4().hex[:8]}"


def generate_class_id() -> str:
    """Generate a short, human-readable class ID.

    Returns:
        6-character alphanumeric string (e.g., 'a3f8d2').

    """
    return uuid.uuid4().hex[:6]


class HostConfig(BaseModel):
    """Minimal configuration provided by the host to generate a scenario.

    This is the starting point - the AI expands these 3 parameters
    into a full wilderness rescue scenario with multiple turns.
    """

    num_participants: int = Field(
        ..., ge=1, le=20, description="Number of participants involved (1-20)"
    )
    activity_type: Literal["canyoneering", "skiing", "hiking"] = Field(
        ..., description="Type of outdoor activity"
    )
    difficulty: Literal["low", "med", "high"] = Field(
        ..., description="Scenario difficulty level"
    )
    class_id: str | None = Field(
        default=None,
        description="Optional class grouping ID (e.g., 'class-2024-wfa')",
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
        description="2-3 multiple choice actions available",
        min_length=2,
        max_length=3,
    )
    is_starting_turn: bool = Field(
        default=False, description="Whether this is the first turn"
    )


class ScenarioDraft(BaseModel):
    """Complete AI-generated wilderness rescue scenario.

    Generated from HostConfig and contains all turns pre-written.
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
        ..., description="All turns in this scenario", min_length=1
    )
    starting_turn_id: int = Field(..., description="ID of the first turn")

    def get_turn(self, turn_id: int) -> ScenarioTurn | None:
        """Get a turn by its ID."""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None


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
