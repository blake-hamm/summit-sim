"""Medical judge for evaluating medical accuracy.

Evaluates ActionResponder outputs for:
- WAS_CORRECT_TREATMENT_GATE: was_correct accurately reflects whether student
  performed treatment without assessment

Treatment without assessment (was_correct should be FALSE):
- Splinting, bandaging, medication, moving patient without vitals/head-to-toe

Assessment without treatment (was_correct should be TRUE):
- Vitals, physical exam, SAMPLE history, identification without intervention

References hidden_truth and hidden_state to determine if treatment was premature.
"""

from mlflow.genai.judges import Judge, make_judge

from summit_sim.judges.utils import (
    JUDGE_MODEL_ENDPOINT,
    get_judge_from_cache,
    set_judge_in_cache,
)

MEDICAL_JUDGE_INSTRUCTIONS = """\
You are evaluating the medical accuracy of an AI-generated response in a \
wilderness first responder training simulation.

Evaluate the trace:
WAS_CORRECT_TREATMENT_GATE: Is was_correct accurate?
- was_correct should be FALSE if student performed treatment \
(splint, bandage, medication, move patient) without assessment
- was_correct should be TRUE for assessment actions (vitals, exam, SAMPLE history)

Reference hidden_truth and hidden_state to determine if treatment was premature.

Trace data: {{ trace }}

Output format: Return True if was_correct is accurate, False otherwise.
"""


def get_medical_judge() -> Judge:
    """Get or create medical judge.

    Medical judge evaluates whether was_correct accurately reflects whether
    the student performed treatment without proper assessment. References
    hidden_truth and hidden_state to determine if treatment was premature.

    Returns:
        Judge instance configured for medical accuracy evaluation

    """
    judge = get_judge_from_cache("medical")
    if judge is None:
        judge = make_judge(
            name="medical-judge",
            instructions=MEDICAL_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=bool,
        )
        set_judge_in_cache("medical", judge)
    return judge
