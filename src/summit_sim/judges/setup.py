"""Judge registration and initialization."""

import logging

from mlflow.genai.scorers import ScorerSamplingConfig

from summit_sim.judges.continuity import get_continuity_judge
from summit_sim.judges.medical import get_medical_judge
from summit_sim.judges.scoring import get_scoring_judge
from summit_sim.judges.structure import get_structure_judge

logger = logging.getLogger(__name__)


def register_trace_judges() -> None:
    """Register trace-level judges for automatic evaluation."""
    # Register all trace-level judges
    judges = [
        ("structure", get_structure_judge()),
        ("scoring", get_scoring_judge()),
        ("medical", get_medical_judge()),
    ]

    for name, judge in judges:
        registered = judge.register(name=f"trace-{name}")
        registered.start(
            sampling_config=ScorerSamplingConfig(sample_rate=1.0),
            filter_string=(
                "tags.graph_type = 'simulation' "
                "AND tags.agent_name = 'action-responder'"
            ),
        )
        logger.info(f"Registered {name} judge for automatic evaluation")


def register_session_judges() -> None:
    """Register session-level judges for automatic evaluation."""
    # Register continuity judge (session scope)
    continuity = get_continuity_judge()
    registered = continuity.register(name="session-continuity")
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
        filter_string="tags.graph_type = 'simulation' ",
    )
    logger.info("Registered continuity judge for automatic evaluation")


def initialize_judges() -> None:
    """Initialize all judges. Call during app startup."""
    register_trace_judges()
    register_session_judges()
    logger.info("Automatic evaluation judges registered")
