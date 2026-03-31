"""Scoring judge for evaluating scoring accuracy and pedagogical quality.

Evaluates ActionResponder outputs for:
- SCORE_MILESTONE_JUSTIFIED: completion_score aligns with PAS rubric progress
- SCORE_NOT_OVER_AWARDED: reasonable score increases (<=0.2 unless bundling)
- FEEDBACK_ACKNOWLEDGES_ACTIONS: feedback specifically mentions student's action

PAS Rubric reference: 0.0=start, 0.2=scene/primary, 0.4=secondary, 0.6=treatment,
0.8=extended care, 1.0=evacuation
"""

from mlflow.genai.judges import Judge, make_judge

from summit_sim.judges.utils import (
    JUDGE_MODEL_ENDPOINT,
    get_judge_from_cache,
    set_judge_in_cache,
)

SCORING_JUDGE_INSTRUCTIONS = """\
You are evaluating the scoring accuracy and pedagogical quality of an \
AI-generated response in a wilderness first responder training simulation.

Context:
- PAS Rubric: 0.0=start, 0.2=scene/primary, 0.4=secondary, 0.6=treatment, \
0.8=extended care, 1.0=evacuation
- Student transcript shows all previous actions

Evaluate the trace:
1. SCORE_MILESTONE_JUSTIFIED: Does completion_score align with the PAS rubric \
based on cumulative actions?
2. SCORE_NOT_OVER_AWARDED: Is the score increase from previous turn reasonable \
(<=0.2 unless explicit bundling)?
3. FEEDBACK_ACKNOWLEDGES_ACTIONS: Does feedback specifically mention \
the student's action?

Input: {{ inputs }}

Output: {{ outputs }}

Return boolean value for criteria
"""


def get_scoring_judge(judge_model: str = JUDGE_MODEL_ENDPOINT) -> Judge:  # noqa: ARG001
    """Get or create scoring judge.

    Scoring judge evaluates the accuracy of completion_score against the
    PAS (Patient Assessment System) rubric and pedagogical quality of feedback.

    Returns:
        Judge instance configured for scoring evaluation

    """
    judge = get_judge_from_cache("scoring")
    if judge is None:
        judge = make_judge(
            name="scoring-judge",
            instructions=SCORING_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=dict[str, bool],
        )
        set_judge_in_cache("scoring", judge)
    return judge
