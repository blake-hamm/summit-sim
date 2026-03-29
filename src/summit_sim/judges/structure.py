"""Structure judge for evaluating response structure and formatting.

Evaluates ActionResponder outputs for:
- SCORE_IN_RANGE: completion_score between 0.0 and 1.0
- QUESTION_IN_NARRATIVE_ONLY: narrative ends with question, feedback has no questions
- FEEDBACK_NO_HARSH_LANGUAGE: encouraging tone without harsh corrections
- NARRATIVE_LENGTH: 3-5 sentences in narrative_text
"""

from mlflow.genai.judges import Judge, make_judge

from summit_sim.judges.utils import (
    JUDGE_MODEL_ENDPOINT,
    get_judge_from_cache,
    set_judge_in_cache,
)

STRUCTURE_JUDGE_INSTRUCTIONS = """\
You are evaluating the structure and formatting of an AI-generated response \
in a wilderness first responder training simulation.

Evaluate the trace against these criteria:

1. SCORE_IN_RANGE: Is completion_score between 0.0 and 1.0?
2. QUESTION_IN_NARRATIVE_ONLY: Does narrative_text end with an open question? \
Does feedback contain NO questions?
3. FEEDBACK_NO_HARSH_LANGUAGE: Is the feedback encouraging and constructive \
without harsh corrections?
4. NARRATIVE_LENGTH: Is narrative_text between 3-5 sentences?

Trace data: {{ trace }}

Output format (JSON with boolean values only):
{
    "score_in_range": true,
    "question_in_narrative_only": true,
    "feedback_no_harsh_language": true,
    "narrative_length": false
}

Return only boolean values for each criterion. No reasons or explanations needed.
"""


def get_structure_judge() -> Judge:
    """Get or create structure judge.

    Structure judge evaluates formatting and structural compliance of
    ActionResponder outputs against defined criteria.

    Returns:
        Judge instance configured for structure evaluation

    """
    judge = get_judge_from_cache("structure")
    if judge is None:
        judge = make_judge(
            name="structure-judge",
            instructions=STRUCTURE_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=dict[str, bool],
        )
        set_judge_in_cache("structure", judge)
    return judge
