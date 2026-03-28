"""Action Responder agent for dynamic simulation turns."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from summit_sim.agents.utils import setup_agent_and_prompts
from summit_sim.schemas import DynamicTurnResult, ScenarioDraft

logger = logging.getLogger(__name__)


@dataclass
class TurnContext:
    """Context for a single simulation turn."""

    hidden_state: str
    scene_state: str
    transcript_history: list[dict]
    turn_count: int
    max_turns: int


AGENT_NAME = "action-responder"

SYSTEM_PROMPT = (
    "You are an expert wilderness first aid instructor evaluating "
    "student actions in real-time.\n\n"
    "Your task is to evaluate a student's free-text action, provide "
    "feedback, generate the next narrative, and update the simulation "
    "state. This happens dynamically - each student action drives the "
    "simulation forward.\n\n"
    "CRITICAL CONSTRAINT: You may ONLY worsen the patient's condition "
    "if the student's action was incorrect (was_correct=False). If the "
    "action was correct, the patient should stabilize or remain the "
    "same - never worsen.\n\n"
    "Your output must follow this exact order (enforced by schema):\n\n"
    "1. EVALUATION (first):\n"
    "   - was_correct: True if action followed proper wilderness first "
    "aid protocols\n"
    "   - completion_score: 0.0-1.0 progress toward scenario resolution "
    "(1.0 = complete evacuation/stabilization)\n"
    "   - is_complete: True only if evacuation is complete OR patient "
    "is fully stabilized with definitive care plan\n"
    "   - feedback: Specific, educational feedback explaining why the "
    "action was correct or incorrect. Be constructive and reference "
    "WFR protocols.\n\n"
    "2. NARRATIVE GENERATION (second):\n"
    "   - narrative_text: Vivid description of what happens next "
    "(3-5 sentences)\n"
    "   - Constraint: If was_correct=False, show negative consequences "
    "(worsening vitals, complications)\n"
    "   - If was_correct=True, show stabilization or improvement\n"
    "   - End with a hook that invites the next student action\n"
    "   - Maintain continuity with previous narrative\n\n"
    "3. STATE EVOLUTION (third):\n"
    "   - updated_hidden_state: Complete replacement of hidden medical "
    "state\n"
    "     * Include all relevant medical details: vitals, injuries, "
    "medications, time elapsed\n"
    "     * Reference previous state but show changes based on student "
    "action\n"
    '     * Example: "Patient remains stable. Fracture immobilized with '
    "SAM splint.\n"
    "       Distal pulse strong. No signs of compartment syndrome. "
    "Given 600mg ibuprofen\n"
    "       10 minutes ago. Pain level reduced from 8/10 to 5/10. "
    "Blood pressure 135/85,\n"
    "       heart rate 82, respiratory rate 14. No changes to initial "
    'mechanism."\n'
    "   - updated_scene_state: Complete replacement of scene/"
    "environmental state\n"
    "     * Include: weather changes, time elapsed, resource usage, "
    "group dynamics\n"
    "     * Reference previous state but show natural evolution\n"
    '     * Example: "45 minutes elapsed since initial assessment. '
    "Weather remains clear,\n"
    "       temperature dropped to 62F. Sunset in 2 hours 15 minutes. "
    "Group has set up\n"
    "       shelter and established radio contact with rescue. Morale "
    "improved - patient\n"
    "       is responsive and comfortable. Used 1 SAM splint, 2 "
    "bandages, 4 ibuprofen\n"
    '       from first aid kit."\n\n'
    "MEDICAL ACCURACY REQUIREMENTS:\n"
    "- Follow current WFR protocols and best practices\n"
    "- Consequences must be realistic for the specific injury/condition\n"
    "- Time progression should be realistic (treatment takes time, "
    "vitals change gradually)\n"
    "- Environmental factors should affect patient condition "
    "appropriately\n\n"
    "NARRATIVE QUALITY:\n"
    "- Be immersive and descriptive\n"
    "- Show, don't just tell (describe what the student sees/hears)\n"
    "- Maintain tension without being overly dramatic\n"
    "- Educational but engaging\n\n"
    "CONTINUITY:\n"
    "- Reference previous actions and their effects\n"
    "- Show realistic progression of time and condition\n"
    "- Maintain consistency with established facts\n"
    "- Build toward scenario resolution naturally"
)

USER_PROMPT_TEMPLATE = (
    "Evaluate the following student action in the wilderness rescue "
    "scenario:\n\n"
    "=== SCENARIO CONTEXT ===\n"
    "Title: {{title}}\n"
    "Setting: {{setting}}\n"
    "Patient Summary: {{patient_summary}}\n"
    "Hidden Truth: {{hidden_truth}}\n"
    "Learning Objectives: {{learning_objectives}}\n\n"
    "=== CURRENT STATE ===\n"
    "Hidden State (medical details known only to AI):\n"
    "{{hidden_state}}\n\n"
    "Scene State (environmental conditions):\n"
    "{{scene_state}}\n\n"
    "=== TRANSCRIPT HISTORY (last 3-5 turns) ===\n"
    "{{transcript_history}}\n\n"
    "=== STUDENT ACTION ===\n"
    '"{{student_action}}"\n\n'
    "=== TURN COUNT ===\n"
    "Turn {{turn_count}} of maximum {{max_turns}} turns\n\n"
    "Evaluate this action and generate the next turn following the "
    "schema requirements.\n"
    "Remember:\n"
    "- Was the action medically correct per WFR protocols?\n"
    "- Only worsen condition if action was incorrect\n"
    "- Update both states completely while maintaining continuity\n"
    "- Consider turn count (approaching max may influence completion "
    "decisions)"
)


async def process_action(
    student_action: str,
    scenario: ScenarioDraft,
    context: TurnContext,
) -> DynamicTurnResult:
    """Process a student action and generate the next simulation turn."""
    logger.info(
        "Processing student action: turn=%d/%d, action_length=%d",
        context.turn_count,
        context.max_turns,
        len(student_action),
    )

    agent, user_prompt = setup_agent_and_prompts(
        agent_name=AGENT_NAME,
        output_type=DynamicTurnResult,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
    )

    # Format transcript history as string
    history_str = _format_transcript_history(context.transcript_history)

    prompt = str(
        user_prompt.format(
            title=scenario.title,
            setting=scenario.setting,
            patient_summary=scenario.patient_summary,
            hidden_truth=scenario.hidden_truth,
            learning_objectives=", ".join(scenario.learning_objectives),
            hidden_state=context.hidden_state,
            scene_state=context.scene_state,
            transcript_history=history_str,
            student_action=student_action,
            turn_count=context.turn_count,
            max_turns=context.max_turns,
        )
    )

    result = await agent.run(prompt)
    logger.info(
        "Action processed: was_correct=%s, is_complete=%s, completion_score=%.2f",
        result.output.was_correct,
        result.output.is_complete,
        result.output.completion_score,
    )
    return result.output


def _format_transcript_history(transcript_history: list[dict]) -> str:
    """Format transcript history for prompt context."""
    if not transcript_history:
        return "No previous actions (initial turn)."

    lines = []
    for i, entry in enumerate(transcript_history[-5:], 1):
        lines.append(f"\nTurn {i}:")
        lines.append(f"  Student: {entry.get('action', 'N/A')}")
        lines.append(f"  Feedback: {entry.get('feedback', 'N/A')[:100]}...")
        lines.append(f"  Result: {entry.get('narrative', 'N/A')[:150]}...")

    return "\n".join(lines)
