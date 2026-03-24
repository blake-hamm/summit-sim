"""LangGraph state definitions for simulation workflow."""

from typing import Annotated, Any

from typing_extensions import TypedDict

from summit_sim.schemas import ChoiceOption, DebriefReport, ScenarioDraft


def append_reducer(left: list, right: list) -> list:
    """Append right items to left list for LangGraph state merging."""
    return left + right


class TranscriptEntry(TypedDict):
    """Single entry in simulation transcript with full context.

    Captures complete information about a turn for debrief analysis.
    """

    turn_id: int
    turn_narrative: str
    choice_id: str
    choice_description: str
    was_correct: bool
    feedback: str
    learning_moments: list[str]
    next_turn_id: int | None


class SimulationState(TypedDict):
    """LangGraph state for simulation workflow.

    Maintains all state needed for the cyclic simulation graph,
    including the scenario, current position, and accumulated history.
    Uses append_reducer for list fields to accumulate history across turns.
    """

    scenario_draft: ScenarioDraft
    current_turn_id: int
    transcript: Annotated[list[TranscriptEntry], append_reducer]
    is_complete: bool
    key_learning_moments: Annotated[list[str], append_reducer]
    last_selected_choice: ChoiceOption | None
    simulation_result: Any
    scenario_id: str
    class_id: str | None
    debrief_report: DebriefReport | None
