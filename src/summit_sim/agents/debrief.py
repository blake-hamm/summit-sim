"""Debrief agent for analyzing completed student simulations."""

from __future__ import annotations

from summit_sim.agents.config import get_agent
from summit_sim.graphs.state import TranscriptEntry
from summit_sim.schemas import DebriefReport, ScenarioDraft

DEBRIEF_SYSTEM_PROMPT = """\
You are an expert wilderness first aid educator analyzing a student's \
simulation performance.

Your task is to review the complete simulation transcript and provide a \
constructive, learning-focused debrief.

Analysis guidelines:
1. Review each turn's choice and the AI feedback provided
2. Identify patterns in decision-making
3. Tally correct vs incorrect choices for scoring
4. Highlight both strengths and areas for improvement
5. Provide specific, actionable recommendations

Scoring:
- Calculate final_score as: (number of correct choices / total turns) * 100
- Determine completion_status: "pass" if final_score >= 70, "fail" otherwise

Tone: Encouraging but honest. Focus on learning, not grading."""

DEBRIEF_USER_PROMPT_TEMPLATE = """\
Analyze this completed wilderness rescue simulation and generate a debrief report.

SCENARIO INFORMATION:
{scenario_context}

SCENARIO ID: {scenario_id}

SIMULATION TRANSCRIPT ({total_turns} turns):
{{'=' * 50}}
{transcript_summary}
{{'=' * 50}}

STATISTICS:
- Total turns: {total_turns}
- Correct choices: {correct_count}
- Incorrect choices: {incorrect_count}
- Calculated score: {score:.1f}%
- Pass threshold: 70%

Provide a comprehensive debrief report following the schema requirements.
The completion_status should be "pass" if score >= 70, otherwise "fail".
The final_score should match the calculated score above."""


async def generate_debrief(
    transcript: list[TranscriptEntry],
    scenario_draft: ScenarioDraft,
    scenario_id: str,
) -> DebriefReport:
    """Generate debrief report from completed simulation.

    Args:
        transcript: Complete simulation transcript
        scenario_draft: Original scenario for context
        scenario_id: Unique scenario identifier

    Returns:
        Structured debrief report with score and analysis

    """
    agent = get_agent(
        agent_name="debrief",
        output_type=DebriefReport,
        system_prompt=DEBRIEF_SYSTEM_PROMPT,
        reasoning_effort="medium",
    )

    prompt = _build_debrief_prompt(transcript, scenario_draft, scenario_id)

    result = await agent.run(prompt)
    return result.output


def _build_debrief_prompt(
    transcript: list[TranscriptEntry],
    scenario_draft: ScenarioDraft,
    scenario_id: str,
) -> str:
    """Build prompt for debrief agent."""
    score = calculate_score(transcript)
    total_turns = len(transcript)
    correct_count = sum(1 for e in transcript if e["was_correct"])
    incorrect_count = total_turns - correct_count

    scenario_context = _format_scenario_context(scenario_draft)
    transcript_summary = _format_transcript_summary(transcript)

    return DEBRIEF_USER_PROMPT_TEMPLATE.format(
        scenario_context=scenario_context,
        scenario_id=scenario_id,
        total_turns=total_turns,
        transcript_summary=transcript_summary,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        score=score,
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
        status = "CORRECT" if entry["was_correct"] else "INCORRECT"
        lines.append(
            f"Turn {entry['turn_id']}: {status}\n"
            f"  Choice: {entry['choice_description']}\n"
            f"  Feedback: {entry['feedback']}"
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
    correct_choices = sum(1 for entry in transcript if entry.get("was_correct", False))

    return (correct_choices / total_turns) * 100
