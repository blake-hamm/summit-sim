"""LangGraph state definitions for simulation workflow."""

from pydantic import BaseModel, Field

from summit_sim.schemas import ScenarioDraft


class TranscriptEntry(BaseModel):
    """Single entry in simulation transcript with full context.

    Captures complete information about a turn for debrief analysis.
    """

    turn_id: int = Field(..., description="ID of the turn")
    turn_narrative: str = Field(..., description="Narrative text shown to student")
    choice_id: str = Field(..., description="ID of the choice selected")
    choice_description: str = Field(..., description="Description of selected choice")
    feedback: str = Field(..., description="AI feedback on the choice")
    learning_moments: list[str] = Field(
        default_factory=list, description="Educational insights from this turn"
    )
    next_turn_id: int | None = Field(
        default=None, description="Next turn ID (null if scenario ends)"
    )


class AppState(BaseModel):
    """LangGraph state for simulation workflow.

    Maintains all state needed for the cyclic simulation graph,
    including the scenario, current position, and accumulated history.
    """

    scenario_draft: ScenarioDraft = Field(
        ..., description="Complete generated scenario with all turns"
    )
    current_turn_id: int = Field(..., description="Current turn the student is on")
    transcript: list[TranscriptEntry] = Field(
        default_factory=list, description="History of all turns and choices"
    )
    is_complete: bool = Field(
        default=False, description="Whether simulation has reached conclusion"
    )
    key_learning_moments: list[str] = Field(
        default_factory=list, description="Accumulated learning moments from all turns"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
