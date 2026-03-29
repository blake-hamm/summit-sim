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
You are an expert wilderness first responder instructor analyzing a student's \
simulation performance.

Your task is to review the complete simulation transcript and provide a \
constructive, expert-level debrief focused entirely on the quality of the \
student's clinical reasoning.

Analysis approach:
- Read the student's typed actions as a real instructor would read their notes
- Compare their reasoning against the hidden medical truth (ground truth)
- Evaluate across WFR dimensions: Scene Safety, PAS Assessment completeness, \
  Treatment decisions, Evacuation timing
- Identify specific moments where their text shows good or poor clinical judgment
- Highlight premature interventions, missed assessment steps, or excellent synthesis

Tone: Expert but encouraging. Make the student feel like a seasoned WFR \
instructor personally reviewed their specific inputs."""


USER_PROMPT_TEMPLATE = """\
Analyze this completed wilderness rescue simulation and generate a debrief report.

SCENARIO INFORMATION:
{{scenario_context}}

SCENARIO ID: {{scenario_id}}

HIDDEN MEDICAL TRUTH (ground truth):
{{hidden_state}}

SIMULATION TRANSCRIPT ({{total_turns}} turns):
{{transcript_summary}}

Provide a comprehensive qualitative debrief analyzing the student's reasoning \
against the hidden medical truth. Focus on their clinical decision-making \
process, not binary right/wrong tallies."""


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
    total_turns = len(transcript)

    scenario_context = _format_scenario_context(scenario_draft)
    transcript_summary = _format_transcript_summary(transcript)

    prompt_result = user_prompt.format(
        scenario_context=scenario_context,
        scenario_id=scenario_id,
        total_turns=total_turns,
        transcript_summary=transcript_summary,
        hidden_state=scenario_draft.hidden_state,
    )
    return str(prompt_result)


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
