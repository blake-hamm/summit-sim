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
based on minimal teacher inputs. You must generate a complete scenario with ALL
turns pre-written.

Guidelines for scenario generation:

1. Create 3-5 turns total for a cohesive learning experience
2. Each turn must have 3-5 multiple choice options:
   - One medically optimal choice (is_correct=true)
   - Others suboptimal but plausible choices (is_correct=false)
   - Choices should be realistic first-responder decisions

3. Turn structure:
   - First turn has turn_id=0
   - Middle turns: branch based on choices
   - Final turns: end the scenario (next_turn_id=null)

4. TURN NARRATIVES: Keep them SHORT and ACTIONABLE
   - Each turn's narrative_text: 1-3 sentences max
   - Focus on the immediate decision point, not scene-setting
   - Students already know the context from previous turns
   - Example GOOD: "Patient grimaces when you touch their forearm.
     Swelling increased. What do you do?"
   - Example BAD: Multi-paragraph description of location, weather,
     patient history, etc.

5. STATE TRACKING: Use hidden_state and scene_state meaningfully
   - hidden_state: Patient condition details that emerge through assessment
     (e.g., "given aspirin 30min ago", "pulse irregularity", "breath sounds diminished")
   - scene_state: Environmental changes (weather deteriorating, new hazards
     appearing, time passing)
   - These add depth for teachers reviewing the scenario

6. SCENARIO-LEVEL CONTENT (can be rich and detailed):
   - Setting: Specific location with environmental details
   - Patient: Age, relevant medical history, visible injuries
   - Hidden truth: The actual medical condition students must discover
   - Learning objectives: 2-3 specific wilderness first aid skills

7. Turn IDs: Sequential integers starting from 0 (e.g., 0, 1, 2, 3, 4).
   Use 0 for the starting turn.

8. Medical accuracy:
   - Base scenarios on common wilderness emergencies
   - Ensure correct symptoms for the condition
   - Treatment options should reflect actual wilderness protocols

Complexity comes from the branching choices and cumulative patient state,
not from lengthy turn narratives."""


USER_PROMPT_TEMPLATE = """\
Generate a wilderness rescue scenario with the following parameters:

Primary Focus (WFR curriculum): {{primary_focus}}
Environment: {{environment}}
Available Personnel: {{available_personnel}}
Evacuation Distance: {{evac_distance}}
Complexity: {{complexity}}

Create a complete scenario with:
- Compelling title and setting appropriate for {{environment}}
- Realistic patient case matching the {{complexity}} complexity
- 3-5 turns with multiple choice decision points
- Medically accurate content
- Clear learning objectives
        - Consider {{available_personnel}} for resource-based decisions
          (litter carries, runners)
        - Account for {{evac_distance}} in stay-and-play vs load-and-go decisions

The scenario should be challenging but educational for wilderness first responders."""


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
