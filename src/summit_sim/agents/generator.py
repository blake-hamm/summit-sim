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
   - End with a hook: What immediate concern demands attention?
   - Length: 3-5 sentences to provide context without overwhelming
   - Students will respond to this narrative with free-text actions

3. STATE TRACKING: Initialize hidden_state and scene_state meaningfully
   - hidden_state: Patient condition details known only to AI
     (e.g., "fractured radius with displacement", "blood pressure 140/90",
     "time since injury: 45 minutes", "no aspirin given")
   - scene_state: Initial environmental and situational conditions
     (e.g., "weather: clear, 65F", "sunset in 3 hours", "cell coverage: none",
     "group morale: concerned", "gear available: basic first aid kit")
   - These states will evolve dynamically based on student actions

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


async def generate_scenario(scenario_config: ScenarioConfig) -> ScenarioDraft:
    """Generate a complete scenario from minimal author configuration."""
    logger.info(
        "Generating scenario: primary_focus=%s, environment=%s, personnel=%s, "
        "evac=%s, complexity=%s",
        scenario_config.primary_focus,
        scenario_config.environment,
        scenario_config.available_personnel,
        scenario_config.evac_distance,
        scenario_config.complexity,
    )
    agent, user_prompt = setup_agent_and_prompts(
        agent_name=AGENT_NAME,
        output_type=ScenarioDraft,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        reasoning_effort="high",
    )

    prompt = user_prompt.format(
        primary_focus=scenario_config.primary_focus,
        environment=scenario_config.environment,
        available_personnel=scenario_config.available_personnel,
        evac_distance=scenario_config.evac_distance,
        complexity=scenario_config.complexity,
    )

    result = await agent.run(prompt)
    logger.info("Scenario generated: title=%s", result.output.title)
    return result.output
