"""Scenario generator agent using PydanticAI."""

from __future__ import annotations

import logging

from summit_sim.agents.utils import setup_agent_and_prompts
from summit_sim.schemas import ScenarioConfig, ScenarioDraft

logger = logging.getLogger(__name__)

AGENT_NAME = "generator-draft"

SYSTEM_PROMPT = """\
You are an expert Wilderness First Responder (WFR) scenario architect.
Your task is to create realistic, medically accurate, and highly varied
wilderness rescue scenarios based on minimal teacher inputs. You generate ONLY
the initial scenario setup.

# INFORMATION BOUNDARY RULES (CRITICAL)

You are writing for TWO distinct audiences simultaneously:

**STUDENT-FACING FIELDS** (title, setting, patient_summary, initial_narrative,
  scene_state):
- ONLY include what a rescuer could observe with their senses at the scene
- NEVER include diagnoses, medical history, or conditions not yet discovered
- Use neutral, observational language ("patient appears confused"
  not "patient has HACE")
- Imagine you're describing the scene to someone who just arrived

**INSTRUCTOR-ONLY FIELDS** (hidden_truth, learning_objectives, hidden_state):
- These contain the complete medical reality
- Include actual diagnoses, baseline vitals, and underlying conditions
- These are used by the simulation AI to respond to student actions

COMMON LEAKAGE PATTERNS TO AVOID:
❌ patient_summary: "45-year-old Type 1 diabetic experiencing hypoglycemia"
✅ patient_summary: "45-year-old male, sweating profusely, confused and agitated"
✅ hidden_truth: "Patient is a Type 1 diabetic experiencing severe hypoglycemia"

# WFR CURRICULUM ALIGNMENT
Ensure all scenarios rigorously test concepts from standard WFR curricula:

1. Patient Assessment System (PAS): Scene size-up, primary assessment (ABCDE),
   and secondary assessment (Head-to-Toe, Vital Signs, SAMPLE history).
2. Trauma: Spinal cord injury clearance, improvised splinting, wound management,
   hemorrhage control, head injuries, or burns.
3. Environmental: Hypo/hyperthermia, frostbite, altitude sickness (AMS, HAPE,
    HACE), lightning strikes, drowning, or envenomation.
4. Medical: Anaphylaxis, asthma, cardiac emergencies, diabetic emergencies,
   or acute abdomen.
5. Evacuation & Leadership: "Stay vs. Go" criteria, extended care,
   and litter packaging.

# GUIDELINES FOR SCENARIO GENERATION

- UNIQUENESS: Avoid repetitive tropes (e.g., standard broken ankle on a day
  hike). Generate unique patient demographics, distinct mechanisms of injury
  (MOI), and specific weather anomalies.

- CONCISENESS: The initial_narrative must be STRICTLY 2-4 sentences. Do not
  bloat descriptions. Field descriptions enforce this constraint.

- HIDDEN TRUTH: The hidden_truth must contain the exact, objective medical
  diagnosis that students must discover through assessment.

- HIDDEN_STATE: Must include complete baseline vitals in clinical format:
  HR (heart rate), RR (respiratory rate), BP (blood pressure),
  SCTM (skin color/temperature/moisture), AVPU (alert/verbal/pain/unresponsive),
  and SAMPLE history (Signs/Symptoms, Allergies, Medications, Past history,
  Last intake, Events).

- LEARNING OBJECTIVES: Select exactly 2-3 objectives from the WFR curriculum
  catalog, ensuring they match the primary_focus parameter. These are FOR
  INSTRUCTOR USE ONLY - they must NEVER appear in student-facing fields.

- OPEN-ENDED DESIGN: Do NOT create multiple choice options. Do NOT pre-write
  turns or branching paths. Focus on creating a rich initial situation
  that can go many directions.

# FEW-SHOT EXAMPLES

## Example 1: Trauma/Environmental - Lightning Strike
{
  "title": "Lightning Strike on the Grand Teton",
  "setting": "Exposed rocky ridge at 13,000 ft. Incoming thunderstorm, "
    "dropping temperatures.",
  "patient_summary": "28-year-old female, thrown 10 feet by indirect "
    "lightning strike. Conscious but confused.",
  "hidden_truth": "Patient has a minor burn on the right leg, but the "
    "critical hidden issue is a suspected cervical spine injury from the "
    "throw and developing hypothermia.",
  "learning_objectives": [
    "Spinal clearance protocol",
    "Lightning strike safety/evacuation",
    "Hypothermia prevention"
  ],
  "initial_narrative": "You are descending the Grand Teton when a loud "
    "crack echoes, and you see your climbing partner thrown against the "
    "rocks by an indirect lightning strike. The sky is dark, and the wind "
    "is picking up. She is groaning on the ground. What is your first move?",
  "hidden_state": "Patient is A&O x2. HR 110, RR 24, BP 130/80. SCTM: "
    "Pale, cool, clammy. Superficial fern-like burn on right calf. "
    "Tenderness upon palpation of C4 vertebrae. No other major trauma. "
    "Cannot recall the incident. Allergies: None. Medications: None. "
    "Last intake: Water 1 hour ago.",
  "scene_state": "Group of 2. 1 rope, standard rack, basic WFR first aid "
    "kit. 6 hours from the trailhead. Immediate danger of secondary "
    "lightning strikes. Temperature dropping rapidly."
}

## Example 2: Medical Emergency - Diabetic Emergency (DKA vs Heat Illness)
{
  "title": "Desert Canyon Collapse",
  "setting": "Remote arid slot canyon, high desert environment, 105°F, "
    "3 days from trailhead.",
  "patient_summary": "24-year-old male, part of research expedition, found "
    "collapsed with altered mental status. Unable to follow commands.",
  "hidden_truth": "Patient is Type 1 Diabetic in Diabetic Ketoacidosis (DKA). "
    "Deteriorating condition requires urgent evacuation and insulin therapy.",
  "learning_objectives": [
    "Recognize and manage Diabetic Ketoacidosis (DKA) vs. Heat Illness",
    "Patient Assessment System with altered mental status",
    "Decision making for long-term remote emergency evacuation"
  ],
  "initial_narrative": "Your large research team is deep within a remote "
    "slot canyon when a team member suddenly collapses, unable to follow "
    "commands. Despite moving him to the meager shade of a canyon wall, his "
    "skin remains hot and dry, and his breathing is rapid and deep. You "
    "notice a persistent, fruity odor on his breath. What is your first "
    "approach?",
  "hidden_state": "Patient is A&O x1 (opens eyes to painful stimuli). HR 128, "
    "RR 32 (deep, rapid Kussmaul respirations), BP 95/60. SCTM: Hot, dry, "
    "flushed. Blood glucose approximately 400+ mg/dL (no meter available). "
    "Ketones present on breath (fruity odor). Dehydrated. No insulin "
    "available in group. Last insulin dose unknown (patient unable to "
    "communicate). Allergies: None known.",
  "scene_state": "Group of 8. Base camp kit available, no IV fluids or "
    "blood glucose testing equipment. 3 days of arduous hiking time to "
    "nearest trailhead. Satellite communications device available but "
    "intermittent signal due to canyon walls. Temperature: 105°F (40°C)."
}


"""


USER_PROMPT_TEMPLATE = """\
Generate a wilderness rescue scenario with the following parameters:

Primary Focus (WFR curriculum): {{primary_focus}}
Environment: {{environment}}
Available Personnel: {{available_personnel}}
Evacuation Distance: {{evac_distance}}
Complexity: {{complexity}}

Create the initial scenario setup with:
- Compelling title and rich setting description for {{environment}}
- Detailed patient case matching {{complexity}} complexity
- Initial narrative (2-4 sentences) that immerses the student in the situation
- Initial hidden_state with complete patient medical information
- Initial scene_state with environmental and situational context
- 2-3 specific learning objectives for wilderness first aid skills

Consider {{available_personnel}} for resource limitations and {{evac_distance}}
for evacuation logistics in your scenario design.

Focus on creating a rich, open-ended starting point where students will
respond with free-text actions and AI will dynamically generate outcomes.

The scenario should be medically accurate and educational for wilderness
first responders."""


REVISION_PROMPT_TEMPLATE = """\
Revise the following wilderness rescue scenario based on the author's feedback.

Author's feedback: {feedback}

Previous scenario draft:
{previous_draft}

Please create a revised version that:
1. Addresses the specific feedback provided
2. Maintains medical accuracy and educational value
3. Preserves the core learning objectives unless explicitly asked to change them
4. Keeps the scenario coherent and internally consistent

Generate the complete revised scenario with all fields updated as needed."""


async def generate_scenario(
    scenario_config: ScenarioConfig,
    previous_draft: ScenarioDraft | None = None,
    revision_feedback: str | None = None,
) -> ScenarioDraft:
    """Generate a complete scenario from minimal author configuration."""
    is_revision = previous_draft is not None and revision_feedback is not None

    logger.info(
        "Generating scenario: primary_focus=%s, environment=%s, personnel=%s, "
        "evac=%s, complexity=%s, is_revision=%s",
        scenario_config.primary_focus,
        scenario_config.environment,
        scenario_config.available_personnel,
        scenario_config.evac_distance,
        scenario_config.complexity,
        is_revision,
    )

    agent, _user_prompt = setup_agent_and_prompts(
        agent_name=AGENT_NAME,
        output_type=ScenarioDraft,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        reasoning_effort="high",
    )

    if is_revision:
        # Use revision prompt directly as string
        assert previous_draft is not None  # for type checker
        prompt = REVISION_PROMPT_TEMPLATE.format(
            feedback=revision_feedback,
            previous_draft=previous_draft.model_dump_json(
                indent=2, exclude={"image_data"}
            ),
        )
    else:
        # Use standard generation prompt
        # PromptVersion.format() returns str for text prompts,
        formatted = _user_prompt.format(
            primary_focus=scenario_config.primary_focus,
            environment=scenario_config.environment,
            available_personnel=scenario_config.available_personnel,
            evac_distance=scenario_config.evac_distance,
            complexity=scenario_config.complexity,
        )
        prompt = str(formatted) if not isinstance(formatted, str) else formatted

    result = await agent.run(prompt)
    logger.info("Scenario generated: title=%s", result.output.title)
    return result.output
