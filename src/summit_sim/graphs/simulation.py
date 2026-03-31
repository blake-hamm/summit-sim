"""LangGraph workflow for dynamic simulation orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import mlflow
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from mlflow.entities import SpanType

from summit_sim.agents.action_responder import (
    AGENT_NAME as ACTION_RESPONDER_AGENT_NAME,
)
from summit_sim.agents.action_responder import (
    ActionRequest,
    ActionResponse,
    action_response_agent,
)
from summit_sim.agents.debrief import (
    AGENT_NAME as DEBRIEF_AGENT_NAME,
)
from summit_sim.agents.debrief import (
    generate_debrief,
)
from summit_sim.schemas import (
    ScenarioDraft,
    TranscriptEntry,
)
from summit_sim.settings import settings

logger = logging.getLogger(__name__)

# Completion threshold - scenario ends when score reaches this value
COMPLETION_THRESHOLD = 0.8

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class SimulationState:
    """LangGraph state for dynamic simulation workflow.

    Maintains all state needed for the free-text simulation graph,
    including scenario context and accumulated history.
    Uses transcript as single source of truth for turn history.
    """

    scenario: ScenarioDraft | None
    transcript: list[TranscriptEntry] = field(default_factory=list)
    turn_count: int = 0
    is_complete: bool = False
    action_response: dict | None = None
    scenario_id: str = ""
    debrief_report: dict | None = None
    hidden_state: str = ""  # Ground truth for AI to reveal from
    current_trace_id: str | None = None  # MLflow trace ID for session correlation

    @classmethod
    def from_graph_result(cls, result: dict[str, Any]) -> "SimulationState":
        """Create state from LangGraph result, filtering extra fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in result.items() if k in valid_fields}
        return cls(**filtered)


def initialize_simulation(state: SimulationState) -> SimulationState:
    """Initialize simulation state from scenario."""
    logger.info(
        "Initializing dynamic simulation: scenario_id=%s",
        state.scenario_id,
    )

    # Initialize states from scenario
    return SimulationState(
        scenario=state.scenario,
        transcript=[],
        turn_count=0,
        is_complete=False,
        action_response=None,
        scenario_id=state.scenario_id,
        debrief_report=None,
        hidden_state=state.scenario.hidden_state,
    )


def present_prompt(state: SimulationState) -> dict:
    """Present current situation to player and wait for free-text action.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the narrative and waits for student text input.
    """
    if state.scenario is None:
        msg = "Cannot present prompt without scenario"
        raise ValueError(msg)

    # Get current narrative (initial or from last action result)
    if state.turn_count == 0:
        current_narrative = state.scenario.initial_narrative
    else:
        result = ActionResponse.model_validate(state.action_response)
        current_narrative = result.narrative_text

    action_data = interrupt(
        {
            "type": "prompt_presented",
            "turn_count": state.turn_count,
            "narrative": current_narrative,
            "is_initial": state.turn_count == 0,
        }
    )

    student_action = action_data.get("action", "").strip()

    if not student_action:
        msg = "Empty student action received"
        raise ValueError(msg)

    # Store the action in a new transcript entry for this turn
    return {
        "transcript": state.transcript
        + [
            TranscriptEntry(
                turn_id=state.turn_count + 1,
                turn_narrative="",  # Will be filled by action result
                student_action=student_action,
                was_correct=False,  # Will be updated by action result
                feedback="",
            )
        ]
    }


@mlflow.trace(span_type="WORKFLOW")
async def process_student_action(
    state: SimulationState, config: RunnableConfig
) -> dict:
    """Process student's free-text action and generate response.

    Calls the ActionResponder agent to evaluate the action,
    generate narrative, and update simulation state.
    """
    if state.scenario is None:
        msg = "Cannot process action without scenario"
        raise ValueError(msg)

    max_turns = settings.max_turns

    # Get the student action from the last transcript entry
    if not state.transcript:
        msg = "No transcript entry found for student action"
        raise ValueError(msg)

    student_action = state.transcript[-1].student_action

    # Extract previous score for progressive tracking
    previous_score = 0.0
    if state.action_response:
        previous_score = state.action_response.get("completion_score", 0.0)

    # Convert transcript entries to dicts for the agent
    transcript_dicts = [
        {
            "turn_id": entry.turn_id,
            "student_action": entry.student_action,
            "turn_narrative": entry.turn_narrative,
            "was_correct": entry.was_correct,
            "feedback": entry.feedback,
        }
        for entry in state.transcript
    ]

    # Build the clean input model for the agent
    input_data = ActionRequest(
        student_action=student_action,
        scenario_title=state.scenario.title,
        scenario_setting=state.scenario.setting,
        patient_summary=state.scenario.patient_summary,
        hidden_truth=state.scenario.hidden_truth,
        learning_objectives=state.scenario.learning_objectives,
        transcript=transcript_dicts,
        previous_score=previous_score,
        turn_count=state.turn_count + 1,
        max_turns=max_turns,
        hidden_state=state.hidden_state,
    )

    result = await action_response_agent(input_data)

    # Programmatic failsafe: ensure score never decreases
    result.completion_score = max(previous_score, result.completion_score)

    # Get current trace info for session correlation
    active_span = mlflow.get_current_active_span()
    current_trace_id = active_span.trace_id if active_span else None

    # Tag trace with session ID for correlation
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id and current_trace_id:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": thread_id},
            tags={
                "session_id": thread_id,
                "scenario_id": state.scenario_id,
                "graph_type": "simulation",
                "agent_name": ACTION_RESPONDER_AGENT_NAME,
                "mlflow_env": settings.mlflow_env,
            },
        )

    # Set span attributes for granular tracking
    if active_span:
        active_span.set_attributes(
            {
                "simulation.turn_count": state.turn_count + 1,
                "agent.name": ACTION_RESPONDER_AGENT_NAME,
                "student.action_length": len(student_action),
                "turn.was_correct": result.was_correct,
                "turn.completion_score": result.completion_score,
                "turn.is_complete": result.completion_score > COMPLETION_THRESHOLD,
            }
        )

    return {
        "action_response": result.model_dump(),
        "current_trace_id": current_trace_id,
    }


def update_simulation_state(state: SimulationState) -> dict:
    """Update simulation state after processing action."""
    result = ActionResponse.model_validate(state.action_response)

    # Update the last transcript entry with results
    if not state.transcript:
        msg = "No transcript entry to update"
        raise ValueError(msg)

    updated_entry = TranscriptEntry(
        turn_id=state.transcript[-1].turn_id,
        turn_narrative=result.narrative_text,
        student_action=state.transcript[-1].student_action,
        was_correct=result.was_correct,
        feedback=result.feedback,
    )

    updated_transcript = state.transcript[:-1] + [updated_entry]

    # Check if scenario is naturally complete or if we've hit max turns
    # is_complete is derived from completion_score (>= 0.7 = 70% threshold)
    next_turn_count = state.turn_count + 1
    is_max_turns_reached = next_turn_count >= settings.max_turns
    is_evacuation_complete = result.completion_score > COMPLETION_THRESHOLD
    is_complete = is_evacuation_complete or is_max_turns_reached

    if is_max_turns_reached and not is_evacuation_complete:
        logger.info(
            "Simulation ending due to max turns: turn=%d, max=%d",
            next_turn_count,
            settings.max_turns,
        )

    logger.debug(
        "Turn processed: turn=%d, correct=%s, complete=%s, score=%.2f",
        next_turn_count,
        result.was_correct,
        is_complete,
        result.completion_score,
    )

    return {
        "transcript": updated_transcript,
        "turn_count": next_turn_count,
        "is_complete": is_complete,
        # hidden_state stays the same (ground truth doesn't change)
    }


@mlflow.trace(span_type=SpanType.AGENT)
async def generate_debrief_report(
    state: SimulationState,
    config: RunnableConfig,
) -> dict:
    """Generate debrief report after simulation completes.

    Calls the Debrief Agent to analyze the complete simulation transcript
    and generate a structured performance report.
    """
    debrief_report = await generate_debrief(
        transcript=state.transcript,
        scenario_draft=state.scenario,
        scenario_id=state.scenario_id,
    )

    # Get current trace info for session correlation
    active_span = mlflow.get_current_active_span()
    current_trace_id = active_span.trace_id if active_span else None

    # Tag trace with session ID for correlation
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id and current_trace_id:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": thread_id},
            tags={
                "session_id": thread_id,
                "scenario_id": state.scenario_id,
                "graph_type": "simulation",
                "agent_name": DEBRIEF_AGENT_NAME,
                "mlflow_env": settings.mlflow_env,
            },
        )

    # Set span attributes for granular tracking
    if active_span:
        active_span.set_attributes(
            {
                "agent.name": DEBRIEF_AGENT_NAME,
            }
        )

    return {
        "debrief_report": debrief_report.model_dump(),
        "current_trace_id": current_trace_id,
    }


def check_simulation_ending(state: SimulationState) -> str:
    """Check if simulation should continue or end.

    Routes the graph to either continue presenting prompts or generate
    debrief based on the is_complete flag.
    """
    if state.is_complete:
        return "generate_debrief"
    return "present_prompt"


def create_simulation_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Create and configure the dynamic simulation LangGraph."""
    workflow = StateGraph(SimulationState)

    workflow.add_node("initialize", initialize_simulation)
    workflow.add_node("present_prompt", present_prompt)
    workflow.add_node("process_student_action", process_student_action)
    workflow.add_node("update_simulation_state", update_simulation_state)
    workflow.add_node("generate_debrief", generate_debrief_report)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "present_prompt")
    workflow.add_edge("present_prompt", "process_student_action")
    workflow.add_edge("process_student_action", "update_simulation_state")

    workflow.add_conditional_edges(
        "update_simulation_state",
        check_simulation_ending,
        {
            "generate_debrief": "generate_debrief",
            "present_prompt": "present_prompt",
        },
    )

    workflow.add_edge("generate_debrief", END)

    return workflow.compile(checkpointer=checkpointer)
