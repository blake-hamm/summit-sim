"""LangGraph workflow for teacher review orchestration.

This module implements a linear LangGraph workflow that:
1. Initializes a teacher session with generated IDs
2. Generates a scenario from teacher configuration
3. Uses interrupt() for human-in-the-loop review and approval
4. Completes with approval status for sharing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from summit_sim.agents.generator import generate_scenario
from summit_sim.graphs.state import TeacherReviewState
from summit_sim.schemas import generate_class_id, generate_scenario_id

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def initialize_teacher_session(state: TeacherReviewState) -> TeacherReviewState:
    """Initialize teacher session with generated IDs.

    Generates scenario_id and class_id, initializes retry_count to 0,
    and creates empty feedback_history.
    """
    return {
        **state,
        "scenario_id": generate_scenario_id(),
        "class_id": generate_class_id(),
        "retry_count": 0,
        "feedback_history": [],
    }


async def generate_scenario_node(state: TeacherReviewState) -> dict:
    """Generate scenario from teacher configuration.

    Calls the scenario generator agent to create a complete scenario
    based on the teacher's configuration parameters.
    """
    scenario = await generate_scenario(state["teacher_config"])
    return {"scenario_draft": scenario}


def present_for_review(state: TeacherReviewState) -> dict:
    """Present scenario for teacher review and wait for approval.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the generated scenario and waits for teacher decision.
    """
    scenario = state["scenario_draft"]

    if scenario is None:
        msg = "No scenario draft available for review"
        raise ValueError(msg)

    choice_data = interrupt(
        {
            "type": "scenario_review",
            "scenario": scenario,
            "scenario_id": state["scenario_id"],
            "class_id": state["class_id"],
        }
    )

    decision = choice_data.get("decision")

    if decision != "approve":
        msg = f"Invalid decision: {decision}. Expected 'approve'"
        raise ValueError(msg)

    # Tag the trace with SME approval
    mlflow.set_tag("sme_approved", "true")

    return {"approval_status": "approved"}


def create_teacher_review_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Create and configure the teacher review LangGraph."""
    workflow = StateGraph(TeacherReviewState)

    workflow.add_node("initialize", initialize_teacher_session)
    workflow.add_node("generate", generate_scenario_node)
    workflow.add_node("review", present_for_review)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)
