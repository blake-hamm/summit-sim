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
  catalog, ensuring they match the primary_focus parameter.

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

## Example 2: Medical Emergency - Anaphylaxis
{
  "title": "Anaphylaxis on the Pine Ridge Trail",
  "setting": "Dense forest canopy, mid-afternoon, warm and humid. "
    "2 miles from trailhead.",
  "patient_summary": "16-year-old male, bee sting to neck 5 minutes ago. "
    "Complains of throat tightness and dizziness.",
  "hidden_truth": "Patient is experiencing anaphylaxis with rapidly "
    "progressing airway compromise. Requires immediate epinephrine "
    "administration.",
  "learning_objectives": [
    "Recognize anaphylaxis",
    "Administer epinephrine auto-injector",
    "Anaphylaxis airway management"
  ],
  "initial_narrative": "Your hiking group stops for a water break when "
    "you hear someone shout. A teenager is clutching his neck and his "
    "face is swelling. He tells you he was just stung by a bee and his "
    "throat feels tight. What do you do first?",
  "hidden_state": "Patient is A&O x3 but anxious. HR 120, RR 28 (labored), "
    "BP 100/70. SCTM: Flushed, warm, diaphoretic. Visible sting site on "
    "right lateral neck with localized swelling. Hoarse voice. Complains "
    "of throat tightness and lightheadedness. No hives visible yet. "
    "Allergies: Unknown. Medications: None. Last intake: Trail mix 30 "
    "minutes ago.",
  "scene_state": "Group of 5 hikers including patient. Standard day "
    "hiking gear. WFR first aid kit with 2 EpiPens available. Cell "
    "service available but spotty. Nearest trailhead 45 minutes downhill. "
    "Weather stable."
}

## Example 3: Mixed/Complicated - Altitude Illness
{
  "title": "HAPE at 14,000 Feet",
  "setting": "High alpine basin, dusk, cold wind. Camp at 14,200 ft after "
    "3-day approach.",
  "patient_summary": "42-year-old male expedition member, progressive "
    "shortness of breath and cough developing over 24 hours. Now producing "
    "pink frothy sputum.",
  "hidden_truth": "Patient has High Altitude Pulmonary Edema (HAPE) with "
    "moderate severity. Requires immediate descent and supplemental oxygen "
    "if available. Deteriorating condition.",
  "learning_objectives": [
    "Assess altitude illness progression",
    "Apply stay-vs-go decision criteria",
    "HAPE recognition and management"
  ],
  "initial_narrative": "It's day 3 of your alpine expedition and one team "
    "member has been struggling to keep up all day. He's been coughing "
    "persistently and just produced pink frothy sputum. He insists he's "
    "just tired from the climb. The summit attempt is scheduled for tomorrow "
    "morning. How do you respond?",
  "hidden_state": "Patient is A&O x3 but fatigued. HR 118, RR 32 (shallow, "
    "rapid), BP 135/85, SpO2 72%. SCTM: Pale, cool, clammy. Productive "
    "cough with pink frothy sputum. Complains of chest tightness and severe "
    "dyspnea at rest. Crackles audible bilaterally on auscultation. No "
    "headache or ataxia. Allergies: None. Medications: Diamox 125mg BID "
    "(started 2 days ago). No relief with rest.",
  "scene_state": "Expedition of 6 with full mountaineering gear. "
    "Supplemental oxygen available (2L cylinder). 2 days from trailhead at "
    "current pace. Current camp at 14,200 ft. Descent to 12,000 ft possible "
    "tonight but difficult in dark. Weather stable but cold. SAT phone "
    "available."
}

Your output provides the foundation for dynamic, interactive learning "
    "where AI responds to each student decision in real-time."""


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
- Initial narrative (3-5 sentences) that immerses the student in the situation
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
            previous_draft=previous_draft.model_dump_json(indent=2),
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
