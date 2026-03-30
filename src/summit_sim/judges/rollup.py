"""Rollup computation for overall quality score.

This module is designed to be used offline in a Jupyter notebook after
simulations complete. MLflow automatic evaluation takes 1-2 minutes for
traces and 5 minutes for sessions, so run this after waiting for evaluation.
"""

import logging

import mlflow

from summit_sim.agents.action_responder import ActionResponse
from summit_sim.judges.medical import get_medical_judge
from summit_sim.judges.scoring import get_scoring_judge
from summit_sim.judges.structure import get_structure_judge
from summit_sim.judges.utils import JUDGE_WEIGHTS
from summit_sim.schemas import RollupResult
from summit_sim.settings import settings

logger = logging.getLogger(__name__)


def compute_rollup_score(session_id: str) -> RollupResult:
    """Compute overall score from all judge assessments in a session.

    Run this in a Jupyter notebook after simulations complete and MLflow
    has had time to evaluate traces (1-2 min) and sessions (5 min inactivity).

    Args:
        session_id: The session ID to compute rollup for

    Returns:
        RollupResult with overall score and breakdown

    """
    client = mlflow.tracking.MlflowClient()

    # Get traces for this session from the main experiment
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if not experiment:
        msg = f"Experiment {settings.mlflow_experiment_name} not found"
        raise ValueError(msg)

    traces = client.search_traces(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.session_id = '{session_id}'",
    )

    # Aggregate criterion results
    criterion_results: dict[str, bool] = {}
    total_score = 0.0

    for trace in traces:
        # Get assessments for this trace
        trace_data = client.get_trace(trace.info.trace_id)
        if hasattr(trace_data, "assessments"):
            for assessment in trace_data.assessments:
                criterion_name = assessment.name
                passed = assessment.value

                if criterion_name in JUDGE_WEIGHTS:
                    criterion_results[criterion_name] = passed
                    if passed:
                        total_score += JUDGE_WEIGHTS[criterion_name]

    # Calculate contributions
    trace_criteria = {
        k: v
        for k, v in criterion_results.items()
        if k not in ["score_monotonic", "narrative_reveals_progressively"]
    }
    session_criteria = {
        k: v
        for k, v in criterion_results.items()
        if k in ["score_monotonic", "narrative_reveals_progressively"]
    }

    trace_contribution = sum(JUDGE_WEIGHTS[k] for k, v in trace_criteria.items() if v)
    session_contribution = sum(
        JUDGE_WEIGHTS[k] for k, v in session_criteria.items() if v
    )

    return RollupResult(
        session_id=session_id,
        overall_score=total_score,
        trace_contribution=trace_contribution,
        session_contribution=session_contribution,
        breakdown=criterion_results,
        total_criteria=len(criterion_results),
        passed_criteria=sum(1 for v in criterion_results.values() if v),
    )


def compute_rollup_for_all_sessions(
    experiment_name: str | None = None,
) -> list[RollupResult]:
    """Compute rollup scores for all sessions in an experiment.

    Use this in a notebook to batch-process all completed simulations.
    """
    # Use default experiment if not specified
    if experiment_name is None:
        experiment_name = settings.mlflow_experiment_name

    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        logger.warning(f"Experiment {experiment_name} not found")
        return []

    # Get all unique session IDs
    traces = client.search_traces(experiment_ids=[experiment.experiment_id])
    session_ids: set[str] = set()
    for trace in traces:
        trace_data = client.get_trace(trace.info.trace_id)
        if hasattr(trace_data, "tags"):
            session_id = trace_data.tags.get("session_id")
            if session_id:
                session_ids.add(session_id)

    # Compute rollup for each session
    results = []
    for session_id in session_ids:
        try:
            rollup = compute_rollup_score(session_id)
            results.append(rollup)
        except Exception as e:
            logger.error(f"Failed to compute rollup for session {session_id}: {e}")

    return results


def compute_judge_score_for_turn(
    action_response: ActionResponse,
) -> dict[str, float | dict[str, bool] | int]:
    """Compute judge score for a single turn (for prompt iteration testing).

    This is used during prompt optimization to evaluate a single turn
    without needing a full session. Runs all trace-level judges.

    Args:
        action_response: The ActionResponse to evaluate

    Returns:
        dict with overall score and criterion breakdown

    """
    # Run all trace-level judges
    structure_judge = get_structure_judge()
    scoring_judge = get_scoring_judge()
    medical_judge = get_medical_judge()

    # Each judge evaluates the turn
    structure_result = structure_judge.evaluate(action_response)
    scoring_result = scoring_judge.evaluate(action_response)
    medical_result = medical_judge.evaluate(action_response)

    # Compute weighted score
    total_score = 0.0
    breakdown: dict[str, bool] = {}

    # Structure criteria (0.25 total) - flat dict format
    for criterion in [
        "score_in_range",
        "question_in_narrative_only",
        "feedback_no_harsh_language",
        "narrative_length",
    ]:
        passed = structure_result.get(criterion, False)
        breakdown[criterion] = passed
        if passed:
            total_score += JUDGE_WEIGHTS[criterion]

    # Scoring criteria (0.45 total) - flat dict format
    for criterion in [
        "score_milestone_justified",
        "score_not_over_awarded",
        "feedback_acknowledges_actions",
    ]:
        passed = scoring_result.get(criterion, False)
        breakdown[criterion] = passed
        if passed:
            total_score += JUDGE_WEIGHTS[criterion]

    # Medical criteria (0.15 total)
    passed = medical_result  # bool
    breakdown["was_correct_treatment_gate"] = passed
    if passed:
        total_score += JUDGE_WEIGHTS["was_correct_treatment_gate"]

    return {
        "overall_score": total_score,
        "trace_contribution": total_score,  # Session criteria not evaluated
        "breakdown": breakdown,
        "total_criteria": len(breakdown),
        "passed_criteria": sum(1 for v in breakdown.values() if v),
    }
