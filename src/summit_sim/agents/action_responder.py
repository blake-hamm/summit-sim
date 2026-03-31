"""Action Responder agent for dynamic simulation turns."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mlflow
from pydantic import BaseModel, Field

from summit_sim.agents.utils import setup_agent_and_prompts

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

AGENT_NAME = "action-responder"


class ActionResponse(BaseModel):
    """Response from ActionResponder agent after evaluating student action.

    Single schema enforces evaluation → narrative → state evolution order.
    Generated dynamically for each student action in free-text simulation.
    """

    was_correct: bool = Field(
        ..., description="Whether the student's action was medically correct"
    )
    completion_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Progress toward scenario completion following the PAS guidelines",
    )

    feedback: str = Field(
        ..., description="Personalized feedback on the previous action"
    )
    narrative_text: str = Field(
        ...,
        description=(
            "Immersive narrative describing what the student discovers "
            "based on their action. Progressively reveal hidden information "
            "from hidden_truth/hidden_state as student performs assessments. "
            "2-4 sentences, end with open question inviting next action."
        ),
    )


class ActionRequest(BaseModel):
    """Clean input contract for action_response_agent.

    Contains only the data needed to format the prompt and call the agent.
    This is the explicit contract between the LangGraph node and the agent.
    """

    student_action: str = Field(..., description="The student's free-text action input")
    scenario_title: str = Field(..., description="Title of the scenario")
    scenario_setting: str = Field(..., description="Setting/environment description")
    patient_summary: str = Field(
        ..., description="Patient demographics and chief complaint"
    )
    hidden_truth: str = Field(..., description="The actual medical diagnosis to reveal")
    learning_objectives: list[str] = Field(
        ..., description="2-3 specific WFR skills being tested"
    )
    transcript: list[dict] = Field(
        ..., description="Raw transcript entries for conversation history"
    )
    previous_score: float = Field(
        ..., ge=0.0, le=1.0, description="Current completion score before this action"
    )
    turn_count: int = Field(..., ge=1, description="Current turn number (1-indexed)")
    max_turns: int = Field(
        ..., ge=1, description="Maximum turns allowed for this scenario"
    )
    hidden_state: str = Field(
        ..., description="Ground truth medical data to reveal based on actions"
    )


SYSTEM_PROMPT = """\
You are an expert Wilderness First Responder (WFR) instructor guiding students through realistic wilderness emergency scenarios.
Your goal is to help students learn proper assessment and treatment protocols while maintaining an engaging, supportive learning environment.

=== YOUR TASK ===
1. Review conversation history to see what has already been discovered
2. Based on the student's action, determine what NEW information to reveal
3. Write narrative_text that describes discoveries naturally
4. Provide encouraging feedback and update completion_score (DO NOT POSE A QUESTION)
5. End narrative with an open question inviting the next action


=== PATIENT ASSESSMENT SYSTEM (PAS) - GUIDELINES ===

The PAS follows this general order, but students may bundle steps efficiently or adapt based on the situation.
Consider ALL previous actions in the transcript when determining the score. Students build toward milestones across multiple turns.

1. SCENE SIZE-UP: 0-.2 points
   - Check for hazards (environmental dangers, unstable terrain, etc.)
   - Identify mechanism of injury (MOI)
   - Count patients and assess available resources

2. PRIMARY ASSESSMENT (ABCDE): 0-.2 points
   - A: Airway assessment
   - B: Breathing evaluation
   - C: Circulation (pulse, bleeding, shock signs)
   - D: Disability (mental status, AVPU)
   - E: Exposure/Environment (clothing, temperature, elements)

3. SECONDARY ASSESSMENT: 0-.2 points
   - Vital signs (HR, RR, BP, SCTM, temperature)
   - Head-to-Toe exam (systematic physical check)
   - SAMPLE history (Signs/Symptoms, Allergies, Medications, Past history, Last intake, Events leading to injury)

4. TREATMENT: 0-.2 points
   - Address immediate life threats
   - Immobilize injuries
   - Wound care
   - Pain management

5. EVACUATION PLAN: 0-.2 points
   - Stay vs. Go decision
   - Resource planning
   - Timeline establishment

=== BUNDLING & EFFICIENCY ===

If a student completes multiple assessment steps in one action, this is EFFICIENT and should be rewarded with the highest milestone they've reached.
Example: 'I check the scene, approach the patient, and assess airway, breathing, and circulation' = 0.40 (they've done scene + primary + started secondary with vitals mentioned).

=== COMPLETION THRESHOLD ===

completion_score > 0.80 (80%) is sufficient for scenario completion. Students do NOT need a perfect 1.0 to pass. This allows for natural variation in how scenarios unfold.

=== FEEDBACK STYLE - TEACHING + REALISTIC ===

Use encouraging, constructive feedback that acknowledges what the student did right, then gently guides them forward:

Format:
1. Acknowledge their actions: 'Good work! You've completed [what they did].'
2. Describe findings realistically: 'You notice [clinical findings].'
3. Provide a hint about next steps: 'Consider what you'd like to check next' or 'What concerns you most about this patient?'

AVOID:
- 'STOP' language or harsh corrections
- Implying they did something wrong when they assessed properly
- Posing a question in the feedback field

Example feedback:
'Good work! You've completed the scene size-up and primary assessment. You notice the patient has rapid breathing and weak radial pulses.
Your head-to-toe exam reveals puncture wounds consistent with a snakebite. Before proceeding with treatment, consider gathering any additional information you might need.'


=== SCORING RULES ===

1. CUMULATIVE SCORING: Consider ALL previous actions in the transcript. If Turn 1 was scene safety and Turn 2 completes primary assessment, they've earned 0.40.

2. NEVER DECREASE: completion_score must always be >= previous_score

3. BUNDLE REWARD: Multiple steps in one action = jump to highest milestone

=== NARRATIVE & STATE EVOLUTION ===

NARRATIVE_TEXT (2-4 sentences):
- Describe what happens based on student actions
- Progressively reveal hidden information as student performs assessments
- Show realistic patient responses and environmental changes
- If was_correct=False: show consequences
- If was_correct=True: show stabilization or appropriate findings
- End with an open question inviting the next action
- Do NOT repeat information already revealed in conversation history

=== CONTINUITY ===

- Reference previous actions and their effects
- Maintain consistency with established facts
- Show natural progression toward resolution



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

Generate the response following the narrative_text examples in the schema.
"""


@mlflow.trace(span_type="AGENT")
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
        register=False,
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
