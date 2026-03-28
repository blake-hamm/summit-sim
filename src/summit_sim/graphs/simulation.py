"""LangGraph workflow for dynamic simulation orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from summit_sim.agents.action_responder import TurnContext, process_action
from summit_sim.graphs.utils import TranscriptEntry
from summit_sim.schemas import DynamicTurnResult, ScenarioDraft
from summit_sim.settings import get_settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class SimulationState:
    """LangGraph state for dynamic simulation workflow.

    Maintains all state needed for the free-text simulation graph,
    including scenario context, evolving states, and accumulated history.
    """

    scenario_draft: dict | None
    turn_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    is_complete: bool = False
    key_learning_moments: list[str] = field(default_factory=list)
    last_student_action: str | None = None
    action_result: dict | None = None
    scenario_id: str = ""
    debrief_report: dict | None = None
    hidden_state: str = ""
    scene_state: str = ""

    @classmethod
    def from_graph_result(cls, result: dict[str, Any]) -> "SimulationState":
        """Create state from LangGraph result, filtering extra fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in result.items() if k in valid_fields}
        return cls(**filtered)


def initialize_simulation(state: SimulationState) -> SimulationState:
    """Initialize simulation state from scenario draft."""
    logger.info(
        "Initializing dynamic simulation: scenario_id=%s",
        state.scenario_id,
    )
    scenario = ScenarioDraft.model_validate(state.scenario_draft)

    # Initialize states from scenario
    return SimulationState(
        scenario_draft=state.scenario_draft,
        turn_count=0,
        transcript=[],
        is_complete=False,
        key_learning_moments=[],
        last_student_action=None,
        action_result=None,
        scenario_id=state.scenario_id,
        debrief_report=None,
        hidden_state=scenario.hidden_state,
        scene_state=scenario.scene_state,
    )


def present_prompt(state: SimulationState) -> dict:
    """Present current situation to player and wait for free-text action.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the narrative and waits for student text input.
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)

    # Get current narrative (initial or from last action result)
    if state.turn_count == 0:
        current_narrative = scenario.initial_narrative
    else:
        result = DynamicTurnResult.model_validate(state.action_result)
        current_narrative = result.narrative_text

    action_data = interrupt(
        {
            "type": "prompt_presented",
            "turn_count": state.turn_count,
            "narrative": current_narrative,
            "scene_state": state.scene_state,
            "is_initial": state.turn_count == 0,
        }
    )

    student_action = action_data.get("action", "").strip()

    if not student_action:
        msg = "Empty student action received"
        raise ValueError(msg)

    return {"last_student_action": student_action}


async def process_player_action(state: SimulationState) -> dict:
    """Process student's free-text action and generate response.

    Calls the ActionResponder agent to evaluate the action,
    generate narrative, and update simulation state.
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    settings = get_settings()
    max_turns = settings.max_turns

    result = await process_action(
        student_action=state.last_student_action or "",
        scenario=scenario,
        context=TurnContext(
            hidden_state=state.hidden_state,
            scene_state=state.scene_state,
            transcript_history=_build_transcript_context(state.transcript),
            turn_count=state.turn_count + 1,
            max_turns=max_turns,
        ),
    )

    return {"action_result": result.model_dump()}


def update_simulation_state(state: SimulationState) -> dict:
    """Update simulation state after processing action."""
    result = DynamicTurnResult.model_validate(state.action_result)

    # Create transcript entry
    transcript_entry = TranscriptEntry(
        turn_id=state.turn_count + 1,
        turn_narrative=result.narrative_text,
        student_action=state.last_student_action or "",
        was_correct=result.was_correct,
        feedback=result.feedback,
        learning_moments=[result.feedback] if result.feedback else [],
    )

    # Check if scenario is naturally complete or if we've hit max turns
    settings = get_settings()
    next_turn_count = state.turn_count + 1
    is_max_turns_reached = next_turn_count >= settings.max_turns
    is_complete = result.is_complete or is_max_turns_reached

    if is_max_turns_reached and not result.is_complete:
        logger.info(
            "Simulation ending due to max turns: turn=%d, max=%d",
            next_turn_count,
            settings.max_turns,
        )

    logger.info(
        "Turn processed: turn=%d, correct=%s, complete=%s, score=%.2f",
        next_turn_count,
        result.was_correct,
        is_complete,
        result.completion_score,
    )

    return {
        "transcript": state.transcript + [transcript_entry],
        "key_learning_moments": state.key_learning_moments + [result.feedback],
        "turn_count": next_turn_count,
        "is_complete": is_complete,
        "hidden_state": result.updated_hidden_state,
        "scene_state": result.updated_scene_state,
        "last_student_action": None,  # Reset for next turn
    }


def _build_transcript_context(transcript: list[TranscriptEntry]) -> list[dict]:
    """Build simplified transcript context for ActionResponder.

    Returns last 5 turns as context for the agent.
    """
    context = []
    for entry in transcript[-5:]:
        context.append(
            {
                "action": entry.student_action,
                "feedback": entry.feedback,
                "narrative": entry.turn_narrative,
                "was_correct": entry.was_correct,
            }
        )
    return context


async def generate_debrief_report(state: SimulationState) -> dict:
    """Generate debrief report after simulation completes.

    Calls the Debrief Agent to analyze the complete simulation transcript
    and generate a structured performance report.
    """
    from summit_sim.agents.debrief import generate_debrief  # noqa: PLC0415

    debrief_report = await generate_debrief(
        transcript=state.transcript,
        scenario_draft=ScenarioDraft.model_validate(state.scenario_draft),
        scenario_id=state.scenario_id,
    )
    return {"debrief_report": debrief_report.model_dump()}


def check_simulation_ending(state: SimulationState) -> str:
    """Check if simulation should continue or end.

    Routes the graph to either continue presenting prompts or generate
    debrief based on the is_complete flag.
    """
    if state.is_complete:
        return "generate_debrief"
    return "present_prompt"


def create_simulation_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Create and configure the dynamic simulation LangGraph."""
    workflow = StateGraph(SimulationState)

    workflow.add_node("initialize", initialize_simulation)
    workflow.add_node("present_prompt", present_prompt)
    workflow.add_node("process_player_action", process_player_action)
    workflow.add_node("update_simulation_state", update_simulation_state)
    workflow.add_node("generate_debrief", generate_debrief_report)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "present_prompt")
    workflow.add_edge("present_prompt", "process_player_action")
    workflow.add_edge("process_player_action", "update_simulation_state")

    workflow.add_conditional_edges(
        "update_simulation_state",
        check_simulation_ending,
        {
            "generate_debrief": "generate_debrief",
            "present_prompt": "present_prompt",
        },
    )

    workflow.add_edge("generate_debrief", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)
