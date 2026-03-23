"""LangGraph state definitions for simulation workflow."""

from typing import Annotated, Any

from typing_extensions import TypedDict

from summit_sim.schemas import ScenarioDraft


def _add(left: list, right: list) -> list:
    """Reducer that appends right to left."""
    return left + right


class TranscriptEntry(TypedDict):
    """Single entry in simulation transcript with full context.

    Captures complete information about a turn for debrief analysis.
    """

    turn_id: int
    turn_narrative: str
    choice_id: str
    choice_description: str
    feedback: str
    learning_moments: list[str]
    next_turn_id: int | None


class AppState(TypedDict):
    """LangGraph state for simulation workflow.

    Maintains all state needed for the cyclic simulation graph,
    including the scenario, current position, and accumulated history.
    """

    scenario_draft: ScenarioDraft
    current_turn_id: int
    transcript: Annotated[list[TranscriptEntry], _add]
    is_complete: bool
    key_learning_moments: Annotated[list[str], _add]
    last_selected_choice: Any
    simulation_result: Any
    class_id: str
