"""LangGraph workflow for teacher orchestration.

This module implements a linear LangGraph workflow that:
1. Initializes a teacher session with generated IDs
2. Generates a scenario from teacher configuration
3. Uses interrupt() for human-in-the-loop review and approval
4. Completes with approval status for sharing
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import mlflow
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from mlflow.entities import AssessmentSource, AssessmentSourceType

from summit_sim.agents.generator import generate_scenario
from summit_sim.graphs.utils import scenario_store
from summit_sim.schemas import (
    ScenarioDraft,
    TeacherConfig,
    generate_class_id,
    generate_scenario_id,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class TeacherState:
    """LangGraph state for teacher workflow.

    Maintains all state needed for the teacher graph,
    including the teacher configuration, generated scenario draft,
    and review metadata.
    """

    teacher_config: dict
    scenario_draft: dict | None = None
    scenario_id: str = ""
    class_id: str = ""
    retry_count: int = 0
    feedback_history: list[str] = field(default_factory=list)
    approval_status: str | None = None
    current_trace_id: str | None = None

    @classmethod
    def from_graph_result(cls, result: dict[str, Any]) -> "TeacherState":
        """Create state from LangGraph result, filtering extra fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in result.items() if k in valid_fields}
        return cls(**filtered)


def initialize_teacher(state: TeacherState) -> TeacherState:
    """Initialize teacher session with generated IDs.

    Generates scenario_id and class_id, initializes retry_count to 0,
    and creates empty feedback_history.
    """
    return TeacherState(
        teacher_config=state.teacher_config,
        scenario_id=generate_scenario_id(),
        class_id=generate_class_id(),
        retry_count=0,
        feedback_history=[],
    )


async def generate_scenario_node(state: TeacherState) -> dict:
    """Generate scenario from teacher configuration.

    Calls the scenario generator agent to create a complete scenario
    based on the teacher's configuration parameters.
    """
    teacher_config = TeacherConfig.model_validate(state.teacher_config)
    scenario = await generate_scenario(teacher_config)
    active_span = mlflow.get_current_active_span()
    current_trace_id = active_span.trace_id
    return {
        "scenario_draft": scenario.model_dump(),
        "current_trace_id": current_trace_id,
    }


def present_for_teacher(state: TeacherState) -> dict:
    """Present scenario for teacher review and wait for approval.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the generated scenario and waits for teacher decision.
    """
    scenario = state.scenario_draft

    if scenario is None:
        msg = "No scenario draft available for review"
        raise ValueError(msg)

    scenario_obj = ScenarioDraft.model_validate(scenario)
    choice_data = interrupt(
        {
            "type": "scenario_review",
            "scenario": scenario_obj,
            "scenario_id": state.scenario_id,
            "class_id": state.class_id,
        }
    )

    decision = choice_data.get("decision")

    if decision != "approve":
        msg = f"Invalid decision: {decision}. Expected 'approve'"
        raise ValueError(msg)

    if current_trace_id := state.current_trace_id:
        mlflow.log_feedback(
            trace_id=current_trace_id,
            name="teacher_approved",
            value=True,
            rationale="Teacher approved",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id=state.scenario_id
            ),
        )
    return {"approval_status": "approved"}


def save_scenario(state: TeacherState) -> dict:
    """Save approved scenario to LangGraph store."""
    if state.scenario_draft and state.scenario_id:
        scenario_store.put(
            ("scenarios",),
            state.scenario_id,
            {"scenario_draft": state.scenario_draft, "class_id": state.class_id},
        )
    return {}


def create_teacher_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Create and configure the teacher LangGraph."""
    workflow = StateGraph(TeacherState)

    workflow.add_node("initialize", initialize_teacher)
    workflow.add_node("generate", generate_scenario_node)
    workflow.add_node("review", present_for_teacher)
    workflow.add_node("save", save_scenario)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", "save")
    workflow.add_edge("save", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    used_store = store if store is not None else scenario_store

    return workflow.compile(checkpointer=checkpointer, store=used_store)
