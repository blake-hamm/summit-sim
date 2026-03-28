"""Scenario generator agent using PydanticAI."""

from __future__ import annotations

import logging

from summit_sim.agents.utils import setup_agent_and_prompts
from summit_sim.schemas import ScenarioConfig, ScenarioDraft

logger = logging.getLogger(__name__)

AGENT_NAME = "generator-draft"

SYSTEM_PROMPT = """\
You are an expert wilderness rescue scenario designer.

Your task is to create a realistic, medically accurate wilderness rescue scenario
based on minimal teacher inputs. You generate ONLY the initial scenario setup -
the turns and narrative progression will be handled dynamically during simulation.

Guidelines for scenario generation:

1. SCENARIO-LEVEL CONTENT (rich and detailed):
   - Title: Compelling and descriptive
   - Setting: Specific location with environmental details
   - Patient Summary: Age, relevant medical history, visible injuries
   - Hidden Truth: The actual medical condition students must discover
   - Learning Objectives: 2-3 specific wilderness first aid skills

2. INITIAL NARRATIVE: Create an immersive opening scene
   - Set the stage: Where are we? What's the environment like?
   - Introduce the patient: What do we see/know immediately?
   - Establish the situation: What's happening right now?
    - End with a direct question that invites immediate action\n"
    "      (e.g., \"What is your first move?\", \"How do you respond?\",\n"
    "      \"What will you do first?\")
   - Length: 3-5 sentences to provide context without overwhelming
   - Students will respond to this narrative with free-text actions

3. STATE TRACKING: Initialize hidden_state and scene_state as narrative strings
   - hidden_state: Complete patient condition description known only to AI.
     Write this as a cohesive narrative paragraph covering all relevant medical details:
     mechanism of injury, vital signs, underlying conditions, medications given, etc.
     Example: "Patient is a 34-year-old with a closed fracture of the left radius with
     dorsal angulation and displacement. Pulse is present but weak distal to injury.
     Blood pressure 140/90, heart rate 88, respiratory rate 16. Patient reports no
     allergies and has not taken any pain medication. Time since injury: 45 minutes."
   - scene_state: Complete environmental and situational description.
     Write this as a cohesive narrative paragraph covering relevant scene details:
      weather, time of day, available resources, group dynamics, evacuation logistics,
      etc.
     Example: "Clear skies, 65°F, light breeze from the west. Approximately 3 hours
     until sunset. Cell phone coverage is unavailable. Group of 4 hikers, morale is
     concerned but stable. Basic first aid kit available with standard supplies.
     Nearest trailhead is 4 hours hike away."
   - These states will be completely replaced each turn based on student actions
    - Maintain continuity: new state descriptions should reference\n"
    "      and build upon previous conditions

4. Medical accuracy:
   - Base scenarios on common wilderness emergencies
   - Ensure correct symptoms for the condition
   - Treatment options should reflect actual wilderness protocols
   - Account for environmental factors in patient presentation

5. Open-ended design:
   - Do NOT create multiple choice options
   - Do NOT pre-write turns or branching paths
   - Focus on creating a rich initial situation that can go many directions
   - The student's free-text actions will drive the narrative

Your output provides the foundation for dynamic, interactive learning where
AI responds to each student decision in real-time."""


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
        prompt = _user_prompt.format(
            primary_focus=scenario_config.primary_focus,
            environment=scenario_config.environment,
            available_personnel=scenario_config.available_personnel,
            evac_distance=scenario_config.evac_distance,
            complexity=scenario_config.complexity,
        )

    result = await agent.run(prompt)
    logger.info("Scenario generated: title=%s", result.output.title)
    return result.output
