"""LangGraph workflow for simulation orchestration.

This module implements a cyclic LangGraph workflow that:
1. Presents turns to students with multiple choice options
2. Uses interrupt() for human-in-the-loop choice selection
3. Calls the Simulation Feedback Agent for personalized feedback
4. Advances through turns until scenario completion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from summit_sim.agents.debrief import generate_debrief
from summit_sim.agents.simulation import process_choice
from summit_sim.graphs.state import SimulationState, TranscriptEntry

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from summit_sim.schemas import ChoiceOption


def initialize_state(state: SimulationState) -> SimulationState:
    """Initialize simulation state from scenario draft.

    Validates that the starting turn ID exists in the scenario.
    """
    scenario = state["scenario_draft"]
    starting_turn = scenario.get_turn(state["current_turn_id"])

    if starting_turn is None:
        msg = f"Starting turn {state['current_turn_id']} not found in scenario"
        raise ValueError(msg)

    return state


def present_turn(state: SimulationState) -> dict:
    """Present current turn to student and wait for choice selection.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the narrative and available choices, then pauses execution
    until the student provides their choice.
    """
    scenario = state["scenario_draft"]
    current_turn = scenario.get_turn(state["current_turn_id"])

    if current_turn is None:
        msg = f"Turn {state['current_turn_id']} not found in scenario"
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

    return {"last_selected_choice": selected_choice}


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
    scenario = state["scenario_draft"]
    current_turn = scenario.get_turn(state["current_turn_id"])
    selected_choice = state["last_selected_choice"]

    if current_turn is None:
        msg = f"Turn {state['current_turn_id']} not found in scenario"
        raise ValueError(msg)

    result = await process_choice(scenario, current_turn, selected_choice)
    return {"simulation_result": result}


def update_state(state: SimulationState) -> dict:
    """Update simulation state after processing choice.

    Appends transcript entry, updates learning moments, and advances
    to the next turn based on the selected choice.
    """
    scenario = state["scenario_draft"]
    current_turn = scenario.get_turn(state["current_turn_id"])
    result = state["simulation_result"]
    selected_choice = result.selected_choice

    if current_turn is None:
        msg = f"Turn {state['current_turn_id']} not found in scenario"
        raise ValueError(msg)

    transcript_entry: TranscriptEntry = {
        "turn_id": current_turn.turn_id,
        "turn_narrative": current_turn.narrative_text,
        "choice_id": selected_choice.choice_id,
        "choice_description": selected_choice.description,
        "was_correct": selected_choice.is_correct,
        "feedback": result.feedback,
        "learning_moments": result.learning_moments,
        "next_turn_id": selected_choice.next_turn_id,
    }

    is_complete = selected_choice.next_turn_id is None
    next_turn_id = selected_choice.next_turn_id

    if next_turn_id is not None:
        next_turn = scenario.get_turn(next_turn_id)
        if next_turn is None:
            msg = f"Next turn {next_turn_id} not found in scenario"
            raise ValueError(msg)

    return {
        "transcript": [transcript_entry],
        "key_learning_moments": result.learning_moments,
        "current_turn_id": (
            next_turn_id if next_turn_id is not None else state["current_turn_id"]
        ),
        "is_complete": is_complete,
    }


async def generate_debrief_node(state: SimulationState) -> dict:
    """Generate debrief report after simulation completes.

    Calls the Debrief Agent to analyze the complete simulation transcript
    and generate a structured performance report.
    """
    debrief_report = await generate_debrief(
        transcript=state["transcript"],
        scenario_draft=state["scenario_draft"],
        scenario_id=state["scenario_id"],
    )
    return {"debrief_report": debrief_report}


def check_completion(state: SimulationState) -> str:
    """Check if simulation should continue or end.

    Routes the graph to either continue presenting turns or generate
    debrief based on the is_complete flag.
    """
    if state["is_complete"]:
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
