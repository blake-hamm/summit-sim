"""MLflow tracing utilities for summit-sim.

This module provides centralized MLflow configuration and session management
for tracing Pydantic AI and LangGraph executions with parent run support.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import mlflow

from summit_sim.settings import settings

if TYPE_CHECKING:
    from collections.abc import Generator

    from summit_sim.schemas import DebriefReport, TeacherConfig


def enable_tracing(run_tracer_inline: bool = True) -> None:
    """Enable MLflow autologging for all integrations.

    Configures MLflow tracking URI and experiment, then enables
    autologging for Pydantic AI and LangGraph (via LangChain).

    Args:
        run_tracer_inline: Enable inline tracer execution for async context
            propagation. Required when using manual tracing decorators
            inside LangGraph nodes with async methods.

    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    mlflow.pydantic_ai.autolog()
    mlflow.langchain.autolog(run_tracer_inline=run_tracer_inline)


def generate_session_name(config: TeacherConfig, phase: str = "sim") -> str:
    """Generate descriptive session name from config.

    Args:
        config: Teacher configuration containing activity type,
            participant count, difficulty, and class_id.
        phase: Phase identifier ("gen" for generation, "sim" for simulation).

    Returns:
        Descriptive session name including class_id for MLflow linking.

    """
    class_id_part = config.class_id if config.class_id else "noclass"
    return (
        f"{phase}-{class_id_part}-{config.activity_type}-"
        f"{config.num_participants}p-{config.difficulty}"
    )


@contextmanager
def summit_session(
    config: TeacherConfig,
    scenario_id: str,
    session_id: str | None = None,
    phase: Literal["gen", "sim"] = "sim",
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Context manager for Summit-Sim sessions with MLflow parent run.

    Creates a parent MLflow run that encompasses all agent traces,
    with session metadata and descriptive naming. Supports both
    generation ("gen") and simulation ("sim") phases.

    Args:
        config: Teacher configuration for naming and metadata.
        scenario_id: Unique scenario identifier for trace linking.
        session_id: Optional session ID. If not provided, generates a UUID.
        phase: Session phase - "gen" for generation, "sim" for simulation.

    Yields:
        Tuple of (session_id, graph_config) for use in LangGraph invocation.

    Example:
        >>> config = TeacherConfig(num_participants=3, activity_type="hiking")
        >>> with summit_session(
        ...     config, scenario_id="scn-abc123", phase="gen"
        ... ) as (session_id, graph_config):
        ...     graph = create_teacher_review_graph()
        ...     state = await graph.ainvoke(initial_state, graph_config)

    """
    session_id = session_id or str(uuid.uuid4())
    session_name = generate_session_name(config, phase=phase)

    with mlflow.start_run(run_name=session_name):
        # Log session-level parameters
        params = {
            "session_id": session_id,
            "scenario_id": scenario_id,
            "phase": phase,
            "activity_type": config.activity_type,
            "num_participants": config.num_participants,
            "difficulty": config.difficulty,
        }
        if config.class_id:
            params["class_id"] = config.class_id
        mlflow.log_params(params)

        # Set tags for easy filtering
        tags = {
            "session_type": phase,
            "scenario_id": scenario_id,
            "activity_type": config.activity_type,
            "difficulty": config.difficulty,
        }
        if config.class_id:
            tags["class_id"] = config.class_id
        mlflow.set_tags(tags)

        # Graph config with thread_id for trace grouping
        graph_config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

        # Return session id and graph config
        yield session_id, graph_config

        # Log successful completion
        mlflow.set_tag("status", "completed")


@contextmanager
def simulation_session(
    config: TeacherConfig,
    scenario_id: str,
    session_id: str | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Context manager for a simulation session with MLflow parent run.

    Convenience wrapper around summit_session for backward compatibility.
    Creates a parent MLflow run that encompasses all agent traces,
    with session metadata and descriptive naming. Automatically handles
    error tracking and logs final session metrics.

    Args:
        config: Teacher configuration for naming and metadata.
        scenario_id: Unique scenario identifier for trace linking.
        session_id: Optional session ID. If not provided, generates a UUID.

    Yields:
        Tuple of (session_id, graph_config) for use in LangGraph invocation.

    Example:
        >>> config = TeacherConfig(num_participants=3, activity_type="hiking")
        >>> with simulation_session(
        ...     config, scenario_id="scn-abc123"
        ... ) as (session_id, graph_config):
        ...     graph = create_simulation_graph()
        ...     state = await graph.ainvoke(initial_state, graph_config)

    """
    yield from summit_session(config, scenario_id, session_id, phase="sim")


def log_simulation_results(
    transcript: list[dict[str, Any]],
    is_complete: bool,
    key_learning_moments: list[str],
    debrief_report: DebriefReport | None = None,
) -> None:
    """Log final simulation results to the current MLflow run.

    Should be called within a simulation_session context.

    Args:
        transcript: List of transcript entries from the simulation.
        is_complete: Whether the simulation completed successfully.
        key_learning_moments: List of key learning moments accumulated.
        debrief_report: Optional debrief report to log final score and status.

    """
    mlflow.log_metrics(
        {
            "total_turns": len(transcript),
            "is_complete": float(is_complete),
            "learning_moments_count": len(key_learning_moments),
        }
    )

    # Log learning moments as artifacts/tags for easy viewing
    if key_learning_moments:
        mlflow.set_tag("learning_moments_count", str(len(key_learning_moments)))

    # Log debrief metrics if report is provided
    if debrief_report is not None:
        mlflow.log_metrics(
            {
                "final_score": debrief_report.final_score,
            }
        )

        tags: dict[str, str] = {
            "pass_fail": debrief_report.completion_status,
        }
        mlflow.set_tags(tags)
