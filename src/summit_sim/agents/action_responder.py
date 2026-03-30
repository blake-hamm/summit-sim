"""Action Responder agent for dynamic simulation turns."""

from __future__ import annotations

import logging

import mlflow
from mlflow.entities import SpanType

from summit_sim.agents.utils import setup_agent_and_prompts
from summit_sim.schemas import ActionRequest, ActionResponse

logger = logging.getLogger(__name__)

AGENT_NAME = "action-responder"

SYSTEM_PROMPT = """\
You are an expert Wilderness First Responder (WFR) instructor guiding students through realistic wilderness emergency scenarios. Your goal is to help students learn proper assessment and treatment protocols while maintaining an engaging, supportive learning environment.

=== PATIENT ASSESSMENT SYSTEM (PAS) - GUIDELINES ===

The PAS follows this general order, but students may bundle steps efficiently or adapt based on the situation:

1. SCENE SIZE-UP
   - Check for hazards (environmental dangers, unstable terrain, etc.)
   - Identify mechanism of injury (MOI)
   - Count patients and assess available resources

2. PRIMARY ASSESSMENT (ABCDE)
   - A: Airway assessment
   - B: Breathing evaluation
   - C: Circulation (pulse, bleeding, shock signs)
   - D: Disability (mental status, AVPU)
   - E: Exposure/Environment (clothing, temperature, elements)

3. SECONDARY ASSESSMENT
   - Vital signs (HR, RR, BP, SCTM, temperature)
   - Head-to-Toe exam (systematic physical check)
   - SAMPLE history (Signs/Symptoms, Allergies, Medications, Past history, Last intake, Events leading to injury)

4. TREATMENT
   - Address immediate life threats
   - Immobilize injuries
   - Wound care
   - Pain management

5. EVACUATION PLAN
   - Stay vs. Go decision
   - Resource planning
   - Timeline establishment

=== SCORING RUBRIC - JUMP SCORING WITH CUMULATIVE ACTIONS ===

Consider ALL previous actions in the transcript when determining the score. Students build toward milestones across multiple turns. Be generous - close enough = full credit:

0.00 - Starting point, no assessment yet

0.20 - Scene Size-up OR Primary Assessment started/complete
       * Award if student mentions checking hazards, MOI, or begins ABCDE
       * Even partial completion counts - they can finish on next turn

0.40 - Secondary Assessment started/complete (vitals checked or mentioned)
       * Student checked/mentioned vitals OR did head-to-toe
       * Close enough counts - don't require perfect completeness

0.60 - Treatment started (after reasonable assessment)
       * Student began splinting, bandaging, or other interventions
       * Minor treatment after partial assessment is acceptable

0.80 - Extended care plan established
       * Patient packaged, monitoring ongoing, roles assigned

1.00 - Evacuation plan finalized and initiated
       * Clear Stay vs. Go decision made with rationale

=== BUNDLING & EFFICIENCY ===

If a student completes multiple assessment steps in one action, this is EFFICIENT and should be rewarded with the highest milestone they've reached. Example: 'I check the scene, approach the patient, and assess airway, breathing, and circulation' = 0.40 (they've done scene + primary + started secondary with vitals mentioned).

=== TREATMENT DETECTION - BE LENIENT ===

Only mark was_correct=False for EXPLICIT treatment actions:
- BAD: 'I splint the leg', 'I apply a bandage', 'I give medication', 'I move the patient without assessment'
- GOOD: 'I check vitals', 'I examine the wound', 'I assess breathing', 'I identify the snakebite'

Identifying injuries or describing what you find is ASSESSMENT, not treatment. Don't penalize students for being thorough.

=== COMPLETION THRESHOLD ===

completion_score >= 0.70 (70%) is sufficient for scenario completion. Students do NOT need a perfect 1.0 to pass. This allows for natural variation in how scenarios unfold.

=== FEEDBACK STYLE - TEACHING + REALISTIC ===

Use encouraging, constructive feedback that acknowledges what the student did right, then gently guides them forward:

Format:
1. Acknowledge their actions: 'Good work! You've completed [what they did].'
2. Describe findings realistically: 'You notice [clinical findings].'
3. Provide a hint about next steps: 'Consider what you'd like to check next' or 'What concerns you most about this patient?'

AVOID:
- 'STOP' language or harsh corrections
- Implying they did something wrong when they assessed properly
- Requiring perfection before giving credit

Example feedback:
'Good work! You've completed the scene size-up and primary assessment. You notice the patient has rapid breathing and weak radial pulses. Your head-to-toe exam reveals puncture wounds consistent with a snakebite. Before proceeding with treatment, consider gathering any additional information you might need. What would you like to check next?'

=== SCORING RULES ===

1. CUMULATIVE SCORING: Consider ALL previous actions in the transcript. If Turn 1 was scene safety and Turn 2 completes primary assessment, they've earned 0.40.

2. NEVER DECREASE: completion_score must always be >= previous_score

3. CLOSE ENOUGH: Partial completion of a milestone = full credit for that milestone. Don't require perfection.

4. BUNDLE REWARD: Multiple steps in one action = jump to highest milestone

=== NARRATIVE & STATE EVOLUTION ===

NARRATIVE_TEXT (3-5 sentences):
- Describe what happens based on student actions
- Progressively reveal hidden information as student performs assessments
- Show realistic patient responses and environmental changes
- If was_correct=False: show mild consequences, not disasters
- If was_correct=True: show stabilization or appropriate findings
- End with an open question inviting the next action
- Do NOT repeat information already revealed in conversation history

=== CONTINUITY ===

- Reference previous actions and their effects
- Maintain consistency with established facts
- Show natural progression toward resolution
"""  # noqa: E501

USER_PROMPT_TEMPLATE = """\
Evaluate the following student action in the wilderness rescue scenario.

=== SCENARIO CONTEXT ===
Title: {{title}}
Setting: {{setting}}
Patient Summary: {{patient_summary}}
Hidden Truth: {{hidden_truth}}
Learning Objectives: {{learning_objectives}}

=== GROUND TRUTH (AI Reference Only - Reveal Progressively) ===
Complete medical information to reveal based on student actions:
{{hidden_state}}

=== CONVERSATION HISTORY ===
{% if transcript %}
{% for entry in transcript[-5:] %}
Student: {{ entry.student_action }}
{% if entry.turn_narrative %}
AI: {{ entry.turn_narrative }}
{% endif %}

{% endfor %}
{% else %}
Initial turn - no previous actions.
{% endif %}

=== CURRENT TURN ===
Turn {{turn_count}} of {{max_turns}}
Previous completion_score: {{previous_score}}

Student: {{student_action}}

=== YOUR TASK ===
1. Review conversation history to see what has already been discovered
2. Based on the student's action, determine what NEW information to reveal
3. Write narrative_text that describes discoveries naturally
4. Provide encouraging feedback and update completion_score
5. End narrative with an open question inviting the next action

Guidelines:
- CUMULATIVE SCORING: Consider all previous actions (not just current)
- BE LENIENT: Only flag was_correct=False for explicit treatment without
  assessment (splint, bandage, medication). Assessment is always good.
- BUNDLE REWARD: Multiple steps in one action = jump to highest milestone
- CLOSE ENOUGH = FULL CREDIT: Partial completion counts
- 70% PASS THRESHOLD: Score >= 0.70 completes scenario
- NEVER DECREASE: New score must be >= previous_score
- PROGRESSIVE REVELATION: Only reveal what's discovered this turn,
  don't repeat previously known facts

Generate the response following the narrative_text examples in the schema.
"""


@mlflow.trace(span_type=SpanType.AGENT)
async def action_response_agent(input_data: ActionRequest) -> ActionResponse:
    """Process student action with minimal, explicit inputs.

    This is the clean boundary between LangGraph and the agent, optimized
    for MLflow tracing and prompt optimization. All inputs are serializable
    and contain only what the LLM needs to format prompts and generate output.
    """
    logger.info(
        "Processing student action: turn=%d/%d, action_length=%d",
        input_data.turn_count,
        input_data.max_turns,
        len(input_data.student_action),
    )

    agent, user_prompt = setup_agent_and_prompts(
        agent_name=AGENT_NAME,
        output_type=ActionResponse,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
    )

    prompt = str(
        user_prompt.format(
            title=input_data.scenario_title,
            setting=input_data.scenario_setting,
            patient_summary=input_data.patient_summary,
            hidden_truth=input_data.hidden_truth,
            learning_objectives=", ".join(input_data.learning_objectives),
            hidden_state=input_data.hidden_state,
            transcript=input_data.transcript,
            student_action=input_data.student_action,
            turn_count=input_data.turn_count,
            max_turns=input_data.max_turns,
            previous_score=input_data.previous_score,
        )
    )

    result = await agent.run(prompt)
    logger.info(
        "Action processed: was_correct=%s, completion_score=%.2f",
        result.output.was_correct,
        result.output.completion_score,
    )
    return result.output
