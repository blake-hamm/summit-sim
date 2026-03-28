"""Debrief agent for analyzing completed student simulations."""

from __future__ import annotations

import logging

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.model_registry.prompt_version import PromptVersion

from summit_sim.agents.utils import setup_agent_and_prompts
from summit_sim.schemas import DebriefReport, ScenarioDraft, TranscriptEntry

logger = logging.getLogger(__name__)

AGENT_NAME = "debrief"

SYSTEM_PROMPT = """\
You are an expert wilderness first aid educator analyzing a student's \
simulation performance.

Your task is to review the complete simulation transcript and provide a \
constructive, learning-focused debrief.

Analysis guidelines:
1. Review each turn's free-text action and the AI feedback provided
2. Identify patterns in decision-making
3. Tally correct vs incorrect actions for scoring
4. Highlight both strengths and areas for improvement
5. Provide specific, actionable recommendations

Scoring:
- Calculate final_score as: (number of correct actions / total turns) * 100
- Determine completion_status: "pass" if final_score >= 70, "fail" otherwise

Tone: Encouraging but honest. Focus on learning, not grading."""


USER_PROMPT_TEMPLATE = """\
Analyze this completed wilderness rescue simulation and generate a debrief report.

SCENARIO INFORMATION:
{{scenario_context}}

SCENARIO ID: {{scenario_id}}

SIMULATION TRANSCRIPT ({{total_turns}} turns):
{{transcript_summary}}

STATISTICS:
- Total turns: {{total_turns}}
- Correct choices: {{correct_count}}
- Incorrect choices: {{incorrect_count}}
- Calculated score: {{score}}%
- Pass threshold: 70%

Provide a comprehensive debrief report following the schema requirements.
The completion_status should be "pass" if score >= 70, otherwise "fail".
The final_score should match the calculated score above."""


@mlflow.trace(span_type=SpanType.AGENT)
async def generate_debrief(
    transcript: list[TranscriptEntry],
    scenario_draft: ScenarioDraft,
    scenario_id: str,
) -> DebriefReport:
    """Generate debrief report from completed simulation."""
    logger.info(
        "Generating debrief: scenario_id=%s, turns=%d", scenario_id, len(transcript)
    )
    agent, user_prompt = setup_agent_and_prompts(
        agent_name=AGENT_NAME,
        output_type=DebriefReport,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        reasoning_effort="medium",
    )

    prompt = _build_debrief_prompt(transcript, scenario_draft, scenario_id, user_prompt)

    result = await agent.run(prompt)  # type: ignore[arg-type]
    logger.info("Debrief generated: scenario_id=%s", scenario_id)
    return result.output


def _build_debrief_prompt(
    transcript: list[TranscriptEntry],
    scenario_draft: ScenarioDraft,
    scenario_id: str,
    user_prompt: PromptVersion,
) -> str:
    """Build prompt for debrief agent."""
    score = calculate_score(transcript)
    total_turns = len(transcript)
    correct_count = sum(1 for e in transcript if e.was_correct)
    incorrect_count = total_turns - correct_count

    scenario_context = _format_scenario_context(scenario_draft)
    transcript_summary = _format_transcript_summary(transcript)

    return user_prompt.format(
        scenario_context=scenario_context,
        scenario_id=scenario_id,
        total_turns=total_turns,
        transcript_summary=transcript_summary,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        score=f"{score:.1f}",
    )


def _format_scenario_context(scenario_draft: ScenarioDraft) -> str:
    """Format scenario information for prompt."""
    return (
        f"Title: {scenario_draft.title}\n"
        f"Setting: {scenario_draft.setting}\n"
        f"Patient: {scenario_draft.patient_summary}\n"
        f"Learning Objectives: {', '.join(scenario_draft.learning_objectives)}"
    )


def _format_transcript_summary(transcript: list[TranscriptEntry]) -> str:
    """Format transcript entries for prompt."""
    lines = []
    for entry in transcript:
        status = "CORRECT" if entry.was_correct else "INCORRECT"
        lines.append(
            f"Turn {entry.turn_id}: {status}\n"
            f"  Action: {entry.student_action}\n"
            f"  Feedback: {entry.feedback}"
        )
    return "\n".join(lines)


def calculate_score(transcript: list[TranscriptEntry]) -> float:
    """Calculate percentage score from transcript.

    Score = (correct_choices / total_turns) * 100

    Args:
        transcript: List of transcript entries with was_correct field

    Returns:
        Percentage score (0-100)

    """
    if not transcript:
        return 0.0

    total_turns = len(transcript)
    correct_choices = sum(1 for entry in transcript if entry.was_correct)

    return (correct_choices / total_turns) * 100
