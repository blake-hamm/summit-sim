"""LangGraph workflow for author orchestration."""

from __future__ import annotations

import logging
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
    generate_scenario_id,
)

logger = logging.getLogger(__name__)

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
    retry_count: int = 0
    approval_status: str | None = None
    current_trace_id: str | None = None
    author_rating: int | None = None
    revision_feedback: str | None = None

    @classmethod
    def from_graph_result(cls, result: dict[str, Any]) -> "AuthorState":
        """Create state from LangGraph result, filtering extra fields."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in result.items() if k in valid_fields}
        return cls(**filtered)


def initialize_author(state: AuthorState) -> AuthorState:
    """Initialize author session with generated IDs.

    Generates scenario_id and initializes retry_count to 0.
    """
    return AuthorState(
        scenario_config=state.scenario_config,
        scenario_id=generate_scenario_id(),
        retry_count=0,
    )


@mlflow.trace(span_type=SpanType.AGENT)
async def generate_scenario_node(state: AuthorState, config: RunnableConfig) -> dict:
    """Generate scenario from author configuration."""
    logger.info("Generating scenario: retry_count=%d", state.retry_count)
    scenario_config = ScenarioConfig.model_validate(state.scenario_config)

    is_revision = state.revision_feedback is not None
    retry_count = state.retry_count + 1 if is_revision else state.retry_count

    # Get previous draft for revision context
    previous_draft = None
    if is_revision and state.scenario_draft:
        previous_draft = ScenarioDraft.model_validate(state.scenario_draft)

    # Generate scenario with optional revision context
    scenario = await generate_scenario(
        scenario_config,
        previous_draft=previous_draft,
        revision_feedback=state.revision_feedback,
    )

    active_span = mlflow.get_current_active_span()
    current_trace_id = active_span.trace_id if active_span else None

    # Tag trace with session ID for correlation
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id and current_trace_id:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": thread_id},
            tags={"session_id": thread_id},
        )

    # Log revision status to MLflow as trace feedback
    if current_trace_id:
        mlflow.log_feedback(
            trace_id=current_trace_id,
            name="is_revision",
            value=is_revision,
            rationale="Indicates if this generation is a revision of a previous draft",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id=state.scenario_id,
            ),
        )

    return {
        "scenario_draft": scenario.model_dump(),
        "current_trace_id": current_trace_id,
        "retry_count": retry_count,
        "revision_feedback": None,  # Clear after use
    }


def present_for_author(state: AuthorState) -> dict:
    """Present scenario for author review and capture approve/revise decision."""
    scenario = state.scenario_draft

    if scenario is None:
        msg = "No scenario draft available for review"
        raise ValueError(msg)

    scenario_obj = ScenarioDraft.model_validate(scenario)
    logger.info(
        "Scenario ready for review: scenario_id=%s, retry_count=%d",
        state.scenario_id,
        state.retry_count,
    )
    choice_data = interrupt(
        {
            "type": "scenario_review",
            "scenario": scenario_obj,
            "scenario_id": state.scenario_id,
            "retry_count": state.retry_count,
        }
    )

    action = choice_data.get("action")
    feedback = choice_data.get("feedback")

    if action not in ("approve", "revise"):
        msg = f"Invalid action: {action}. Expected 'approve' or 'revise'"
        raise ValueError(msg)

    is_approved = action == "approve"

    # Log approval status to MLflow as trace feedback
    if current_trace_id := state.current_trace_id:
        mlflow.log_feedback(
            trace_id=current_trace_id,
            name="is_approved",
            value=is_approved,
            rationale="Author approval decision for the scenario draft",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id=state.scenario_id,
            ),
        )
        if feedback:
            mlflow.log_feedback(
                trace_id=current_trace_id,
                name="revision_feedback",
                value=feedback,
                rationale="Author requested revision",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id=state.scenario_id,
                ),
            )

    return {
        "approval_status": "approved" if is_approved else "revision_requested",
        "revision_feedback": feedback if not is_approved else None,
    }


def should_retry(state: AuthorState) -> str:
    """Determine if scenario should be regenerated based on revision request.

    Routes to "generate" if revision_feedback is present and retry_count < 3.
    Otherwise routes to "save".
    """
    if state.revision_feedback is not None and state.retry_count < MAX_RETRY_ATTEMPTS:
        return "generate"
    return "save"


def save_scenario(state: AuthorState) -> dict:
    """Save approved scenario to LangGraph store."""
    if state.scenario_draft and state.scenario_id:
        scenario_store.put(
            ("scenarios",),
            state.scenario_id,
            {"scenario_draft": state.scenario_draft},
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
