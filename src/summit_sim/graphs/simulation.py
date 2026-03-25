"""LangGraph workflow for simulation orchestration.

This module implements a cyclic LangGraph workflow that:
1. Presents turns to students with multiple choice options
2. Uses interrupt() for human-in-the-loop choice selection
3. Calls the Simulation Feedback Agent for personalized feedback
4. Advances through turns until scenario completion
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from summit_sim.agents.debrief import generate_debrief
from summit_sim.agents.simulation import process_choice
from summit_sim.graphs.utils import TranscriptEntry
from summit_sim.schemas import ChoiceOption, ScenarioDraft, SimulationResult

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class SimulationState:
    """LangGraph state for simulation workflow.

    Maintains all state needed for the cyclic simulation graph,
    including the scenario, current position, and accumulated history.
    """

    scenario_draft: dict | None
    current_turn_id: int
    transcript: list[TranscriptEntry] = field(default_factory=list)
    is_complete: bool = False
    key_learning_moments: list[str] = field(default_factory=list)
    last_selected_choice: dict | None = None
    simulation_result: dict | None = None
    scenario_id: str = ""
    class_id: str | None = None
    debrief_report: dict | None = None

    @classmethod
    def from_graph_result(cls, result: dict[str, Any]) -> "SimulationState":
        """Create state from LangGraph result, filtering extra fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in result.items() if k in valid_fields}
        return cls(**filtered)


def initialize_state(state: SimulationState) -> SimulationState:
    """Initialize simulation state from scenario draft.

    Validates that the starting turn ID exists in the scenario.
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    starting_turn = scenario.get_turn(state.current_turn_id)

    if starting_turn is None:
        msg = f"Starting turn {state.current_turn_id} not found in scenario"
        raise ValueError(msg)

    return state


def present_turn(state: SimulationState) -> dict:
    """Present current turn to student and wait for choice selection.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the narrative and available choices, then pauses execution
    until the student provides their choice.
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    current_turn = scenario.get_turn(state.current_turn_id)

    if current_turn is None:
        msg = f"Turn {state.current_turn_id} not found in scenario"
        raise ValueError(msg)

    choice_options = [
        {"choice_id": c.choice_id, "description": c.description}
        for c in current_turn.choices
    ]

    choice_data = interrupt(
        {
            "type": "turn_presented",
            "turn_id": current_turn.turn_id,
            "narrative": current_turn.narrative_text,
            "choices": choice_options,
            "scene_state": current_turn.scene_state,
        }
    )

    selected_choice_id = choice_data.get("choice_id")

    selected_choice = None
    for choice in current_turn.choices:
        if choice.choice_id == selected_choice_id:
            selected_choice = choice
            break

    if selected_choice is None:
        msg = f"Invalid choice_id: {selected_choice_id}"
        raise ValueError(msg)

    return {"last_selected_choice": selected_choice.model_dump()}


def _find_choice_by_id(choices: list[ChoiceOption], choice_id: str) -> ChoiceOption:
    """Find a choice by its ID from a list of choices."""
    for choice in choices:
        if choice.choice_id == choice_id:
            return choice
    msg = f"Choice with id '{choice_id}' not found"
    raise ValueError(msg)


async def process_turn(state: SimulationState) -> dict:
    """Process student's choice and generate feedback.

    Calls the Simulation Feedback Agent to generate personalized
    feedback for the student's choice.
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    current_turn = scenario.get_turn(state.current_turn_id)
    selected_choice = ChoiceOption.model_validate(state.last_selected_choice)

    if current_turn is None:
        msg = f"Turn {state.current_turn_id} not found in scenario"
        raise ValueError(msg)

    result = await process_choice(scenario, current_turn, selected_choice)
    return {"simulation_result": result.model_dump()}


def update_state(state: SimulationState) -> dict:
    """Update simulation state after processing choice.

    Appends transcript entry, updates learning moments, and advances
    to the next turn based on the selected choice.
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    current_turn = scenario.get_turn(state.current_turn_id)
    result = SimulationResult.model_validate(state.simulation_result)
    selected_choice = result.selected_choice

    if current_turn is None:
        msg = f"Turn {state.current_turn_id} not found in scenario"
        raise ValueError(msg)

    transcript_entry = TranscriptEntry(
        turn_id=current_turn.turn_id,
        turn_narrative=current_turn.narrative_text,
        choice_id=selected_choice.choice_id,
        choice_description=selected_choice.description,
        was_correct=selected_choice.is_correct,
        feedback=result.feedback,
        learning_moments=result.learning_moments,
        next_turn_id=selected_choice.next_turn_id,
    )

    is_complete = selected_choice.next_turn_id is None
    next_turn_id = selected_choice.next_turn_id

    if next_turn_id is not None:
        next_turn = scenario.get_turn(next_turn_id)
        if next_turn is None:
            msg = f"Next turn {next_turn_id} not found in scenario"
            raise ValueError(msg)

    return {
        "transcript": state.transcript + [transcript_entry],
        "key_learning_moments": state.key_learning_moments + result.learning_moments,
        "current_turn_id": (
            next_turn_id if next_turn_id is not None else state.current_turn_id
        ),
        "is_complete": is_complete,
    }


async def generate_debrief_node(state: SimulationState) -> dict:
    """Generate debrief report after simulation completes.

    Calls the Debrief Agent to analyze the complete simulation transcript
    and generate a structured performance report.
    """
    debrief_report = await generate_debrief(
        transcript=state.transcript,
        scenario_draft=ScenarioDraft.model_validate(state.scenario_draft),
        scenario_id=state.scenario_id,
    )
    return {"debrief_report": debrief_report.model_dump()}


def check_completion(state: SimulationState) -> str:
    """Check if simulation should continue or end.

    Routes the graph to either continue presenting turns or generate
    debrief based on the is_complete flag.
    """
    if state.is_complete:
        return "generate_debrief"
    return "present_turn"


def create_simulation_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Create and configure the simulation LangGraph."""
    workflow = StateGraph(SimulationState)

    workflow.add_node("initialize", initialize_state)
    workflow.add_node("present_turn", present_turn)
    workflow.add_node("process_turn", process_turn)
    workflow.add_node("update_state", update_state)
    workflow.add_node("generate_debrief", generate_debrief_node)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "present_turn")
    workflow.add_edge("present_turn", "process_turn")
    workflow.add_edge("process_turn", "update_state")

    workflow.add_conditional_edges(
        "update_state",
        check_completion,
        {
            "generate_debrief": "generate_debrief",
            "present_turn": "present_turn",
        },
    )

    workflow.add_edge("generate_debrief", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)
