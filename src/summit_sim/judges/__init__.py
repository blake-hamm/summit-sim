"""Judge system for evaluating simulation quality.

This package provides MLflow-based judges for evaluating ActionResponder
outputs across multiple dimensions including structure, scoring, medical
accuracy, and session continuity.

Usage:
    from summit_sim.judges import initialize_judges, compute_rollup_score

    # Initialize during app startup
    initialize_judges()

    # Compute rollup after simulations complete
    result = compute_rollup_score(session_id="session-123")
    print(f"Overall score: {result.overall_score:.2f}")
"""

from summit_sim.judges.continuity import get_continuity_judge
from summit_sim.judges.medical import get_medical_judge
from summit_sim.judges.rollup import (
    compute_judge_score_for_turn,
    compute_rollup_for_all_sessions,
    compute_rollup_score,
)
from summit_sim.judges.scoring import get_scoring_judge
from summit_sim.judges.setup import initialize_judges
from summit_sim.judges.structure import get_structure_judge

__all__ = [
    "initialize_judges",
    "get_structure_judge",
    "get_scoring_judge",
    "get_medical_judge",
    "get_continuity_judge",
    "compute_rollup_score",
    "compute_rollup_for_all_sessions",
    "compute_judge_score_for_turn",
]
