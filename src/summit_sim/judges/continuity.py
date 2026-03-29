"""Continuity judge for evaluating session continuity and progression.

Evaluates across all traces in a session for:
- SCORE_MONOTONIC: completion_score never decreases across turns
- NARRATIVE_REVEALS_PROGRESSIVELY: no facts presented as "new discoveries" are
  repeated across narratives (status updates like "HR is still 110" are OK if
  framed as ongoing)

This is a session-level judge that runs after 5 minutes of inactivity.
"""

from mlflow.genai.judges import Judge, make_judge

from summit_sim.judges.utils import (
    JUDGE_MODEL_ENDPOINT,
    get_judge_from_cache,
    set_judge_in_cache,
)

CONTINUITY_JUDGE_INSTRUCTIONS = """\
You are evaluating the continuity and progression across a complete wilderness \
first responder simulation session.

Evaluate across all traces in the session:

1. SCORE_MONOTONIC: Does completion_score never decrease across turns?
2. NARRATIVE_REVEALS_PROGRESSIVELY: Are facts presented as "new discoveries" \
never repeated? (Status updates like "HR is still 110" are OK if framed as ongoing)

Trace data: {{ trace }}

Output format (JSON):
{
    "score_monotonic": {"passed": bool, "reason": "1-2 sentences"},
    "narrative_reveals_progressively": {"passed": bool, "reason": "1-2 sentences"}
}

Be concise. Cite specific turns where issues occur.
"""


def get_continuity_judge() -> Judge:
    """Get or create continuity judge.

    Continuity judge evaluates progression across a complete simulation session,
    ensuring scores are monotonic and narrative facts are revealed progressively
    without repetition.

    Returns:
        Judge instance configured for continuity evaluation

    """
    judge = get_judge_from_cache("continuity")
    if judge is None:
        judge = make_judge(
            name="continuity-judge",
            instructions=CONTINUITY_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=dict,
        )
        set_judge_in_cache("continuity", judge)
    return judge
