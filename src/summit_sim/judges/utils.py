"""Judge configuration, prompts, and utilities."""

from mlflow.genai.judges import Judge

# Judge weights (must sum to 1.0 across all criteria)
JUDGE_WEIGHTS: dict[str, float] = {
    # Trace-level (Structure Judge)
    "score_in_range": 0.05,
    "question_in_narrative_only": 0.10,
    "feedback_no_harsh_language": 0.05,
    "narrative_length": 0.05,
    # Trace-level (Scoring Judge)
    "score_milestone_justified": 0.20,
    "score_not_over_awarded": 0.15,
    "feedback_acknowledges_actions": 0.05,
    # Trace-level (Medical Judge)
    "was_correct_treatment_gate": 0.15,
    # Session-level (Continuity Judge)
    "score_monotonic": 0.10,
    "narrative_reveals_progressively": 0.10,
}

# Validate weights sum to 1.0
WEIGHT_CHECK: float = 0.001
if abs(sum(JUDGE_WEIGHTS.values()) - 1.0) >= WEIGHT_CHECK:
    msg = f"Weights must sum to 1.0, got {sum(JUDGE_WEIGHTS.values())}"
    raise ValueError(msg)

# OpenRouter direct endpoint for all judges (via LiteLLM)
JUDGE_MODEL_ENDPOINT = "gateway:/openrouter-judge"

# Sampling configuration
TRACE_JUDGE_SAMPLE_RATE: float = 1.0  # 100% in dev
SESSION_JUDGE_SAMPLE_RATE: float = 1.0  # 100% in dev

# Judge instances cache (created on first use)
_judges: dict[str, Judge] = {}


def get_judge_from_cache(name: str) -> Judge | None:
    """Get a judge from the cache if it exists."""
    return _judges.get(name)


def set_judge_in_cache(name: str, judge: Judge) -> None:
    """Store a judge in the cache."""
    _judges[name] = judge
