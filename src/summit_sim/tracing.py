"""MLflow tracing utilities for summit-sim.

This module provides centralized MLflow configuration and session management
for tracing Pydantic AI and LangGraph executions with parent run support.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import mlflow

from summit_sim.settings import settings

if TYPE_CHECKING:
    from collections.abc import Generator

    from summit_sim.schemas import HostConfig


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


def generate_session_name(config: HostConfig, phase: str = "sim") -> str:
    """Generate descriptive session name from config.

    Args:
        config: Host configuration containing activity type,
            participant count, difficulty, and class_id.
        phase: Phase identifier ("gen" for generation, "sim" for simulation).

    Returns:
        Descriptive session name including class_id for MLflow linking.

    """
    return (
        f"{phase}-{config.class_id}-{config.activity_type}-"
        f"{config.num_participants}p-{config.difficulty}"
    )


@contextmanager
def simulation_session(
    config: HostConfig,
    session_id: str | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Context manager for a simulation session with MLflow parent run.

    Creates a parent MLflow run that encompasses all agent traces,
    with session metadata and descriptive naming. Automatically handles
    error tracking and logs final session metrics.

    Args:
        config: Host configuration for naming and metadata.
        session_id: Optional session ID. If not provided, generates a UUID.

    Yields:
        Tuple of (session_id, graph_config) for use in LangGraph invocation.

    Example:
        >>> config = HostConfig(num_participants=3, activity_type="hiking")
        >>> with simulation_session(config) as (session_id, graph_config):
        ...     graph = create_simulation_graph()
        ...     state = await graph.ainvoke(initial_state, graph_config)

    """
    session_id = session_id or str(uuid.uuid4())
    session_name = generate_session_name(config, phase="sim")

    with mlflow.start_run(run_name=session_name):
        # Log session-level parameters
        mlflow.log_params(
            {
                "session_id": session_id,
                "class_id": config.class_id,
                "activity_type": config.activity_type,
                "num_participants": config.num_participants,
                "difficulty": config.difficulty,
            }
        )

        # Set tags for easy filtering
        mlflow.set_tags(
            {
                "session_type": "simulation",
                "class_id": config.class_id,
                "activity_type": config.activity_type,
                "difficulty": config.difficulty,
            }
        )

        # Graph config with thread_id for trace grouping
        graph_config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

        try:
            yield session_id, graph_config
            # Log successful completion
            mlflow.set_tag("status", "completed")
        except Exception:
            # Log failure status before re-raising
            mlflow.set_tag("status", "failed")
            raise


def log_simulation_results(
    transcript: list[dict[str, Any]],
    is_complete: bool,
    key_learning_moments: list[str],
) -> None:
    """Log final simulation results to the current MLflow run.

    Should be called within a simulation_session context.

    Args:
        transcript: List of transcript entries from the simulation.
        is_complete: Whether the simulation completed successfully.
        key_learning_moments: List of key learning moments accumulated.

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
