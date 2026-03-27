"""LangGraph workflow for author orchestration.

This module implements a linear LangGraph workflow that:
1. Initializes an author session with generated IDs
2. Generates a scenario from author configuration
3. Uses interrupt() for human-in-the-loop review and approval
4. Completes with approval status for sharing
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import mlflow
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanType

from summit_sim.agents.generator import generate_scenario
from summit_sim.graphs.utils import scenario_store
from summit_sim.schemas import (
    ScenarioConfig,
    ScenarioDraft,
    generate_class_id,
    generate_scenario_id,
)

MIN_RATING = 1
MAX_RATING = 5
ACCEPTABLE_RATING_THRESHOLD = 3
MAX_RETRY_ATTEMPTS = 3

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass
class AuthorState:
    """LangGraph state for author workflow.

    Maintains all state needed for the author graph,
    including the scenario configuration, generated scenario draft,
    and review metadata.
    """

    scenario_config: dict
    scenario_draft: dict | None = None
    scenario_id: str = ""
    class_id: str = ""
    retry_count: int = 0
    approval_status: str | None = None
    current_trace_id: str | None = None
    author_rating: int | None = None

    @classmethod
    def from_graph_result(cls, result: dict[str, Any]) -> "AuthorState":
        """Create state from LangGraph result, filtering extra fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in result.items() if k in valid_fields}
        return cls(**filtered)


def initialize_author(state: AuthorState) -> AuthorState:
    """Initialize author session with generated IDs.

    Generates scenario_id and class_id, initializes retry_count to 0,
    and creates empty feedback_history.
    """
    return AuthorState(
        scenario_config=state.scenario_config,
        scenario_id=generate_scenario_id(),
        class_id=generate_class_id(),
        retry_count=0,
    )


@mlflow.trace(span_type=SpanType.AGENT)
async def generate_scenario_node(state: AuthorState, config: RunnableConfig) -> dict:
    """Generate scenario from author configuration.

    Calls the scenario generator agent to create a complete scenario
    based on the author's configuration parameters.
    On retry, increments retry_count.
    """
    scenario_config = ScenarioConfig.model_validate(state.scenario_config)

    is_retry = state.author_rating is not None
    retry_count = state.retry_count + 1 if is_retry else 0

    # First generation - let MLflow create a new trace
    scenario = await generate_scenario(scenario_config)
    active_span = mlflow.get_current_active_span()
    current_trace_id = active_span.trace_id if active_span else None

    # Tag trace with session ID for correlation
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id and current_trace_id:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": thread_id},
            tags={"session_id": thread_id},
        )

    return {
        "scenario_draft": scenario.model_dump(),
        "current_trace_id": current_trace_id,
        "retry_count": retry_count,
    }


def present_for_author(state: AuthorState) -> dict:
    """Present scenario for author review and capture rating.

    Uses LangGraph's interrupt() for human-in-the-loop interaction.
    Displays the generated scenario and captures author rating (1-5).
    Automatically retries if rating < 3 (up to 3 attempts).
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
            "retry_count": state.retry_count,
        }
    )

    rating = choice_data.get("rating")

    if (
        rating is None
        or not isinstance(rating, int)
        or not MIN_RATING <= rating <= MAX_RATING
    ):
        msg = f"Invalid rating: {rating}. Expected integer 1-5"
        raise ValueError(msg)

    if current_trace_id := state.current_trace_id:
        mlflow.log_feedback(
            trace_id=current_trace_id,
            name="author_rating",
            value=rating,
            rationale=f"Author rated {rating}/5",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN, source_id=state.scenario_id
            ),
        )

    return {
        "author_rating": rating,
        "approval_status": (
            "approved" if rating >= ACCEPTABLE_RATING_THRESHOLD else "rejected"
        ),
    }


def should_retry(state: AuthorState) -> str:
    """Determine if scenario should be regenerated based on rating.

    Routes to "generate" if rating < 3 and retry_count < 3.
    Otherwise routes to "save".
    """
    rating = state.author_rating
    retry_count = state.retry_count

    if (
        rating is not None
        and rating < ACCEPTABLE_RATING_THRESHOLD
        and retry_count < MAX_RETRY_ATTEMPTS
    ):
        return "generate"
    return "save"


def save_scenario(state: AuthorState) -> dict:
    """Save approved scenario to LangGraph store."""
    if state.scenario_draft and state.scenario_id:
        scenario_store.put(
            ("scenarios",),
            state.scenario_id,
            {"scenario_draft": state.scenario_draft, "class_id": state.class_id},
        )
    return {}


def create_author_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Create and configure the author LangGraph."""
    workflow = StateGraph(AuthorState)

    workflow.add_node("initialize", initialize_author)
    workflow.add_node("generate", generate_scenario_node)
    workflow.add_node("review", present_for_author)
    workflow.add_node("save", save_scenario)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_conditional_edges(
        "review", should_retry, {"generate": "generate", "save": "save"}
    )
    workflow.add_edge("save", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()

    used_store = store if store is not None else scenario_store

    return workflow.compile(checkpointer=checkpointer, store=used_store)
