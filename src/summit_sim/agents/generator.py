"""Scenario generator agent using PydanticAI."""

from __future__ import annotations

import mlflow

from summit_sim.agents.config import get_agent
from summit_sim.schemas import HostConfig, ScenarioDraft
from summit_sim.settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

GENERATOR_SYSTEM_PROMPT = """You are an expert wilderness rescue scenario designer.

Your task is to create a realistic, medically accurate wilderness rescue scenario
based on minimal host inputs. You must generate a complete scenario with ALL
turns pre-written.

Guidelines for scenario generation:

1. Create 3-5 turns total for a cohesive learning experience
2. Each turn must have 2-3 multiple choice options:
   - One medically optimal choice (is_correct=true)
   - One or two suboptimal but plausible choices (is_correct=false)
   - Choices should be realistic first-responder decisions

3. Turn structure:
   - First turn: is_starting_turn=true
   - Middle turns: branch based on choices
   - Final turns: end the scenario (next_turn_id=null)

4. Content requirements:
   - Setting: Specific location with environmental details
   - Patient: Age, relevant medical history, visible injuries
   - Hidden truth: The actual medical condition students must discover
   - Learning objectives: 2-3 specific wilderness first aid skills
   - Narrative: Immersive, medically accurate descriptions

5. Turn IDs should be descriptive (e.g., "initial_assessment",
   "treatment_decision", "evacuation")

6. Medical accuracy:
   - Base scenarios on common wilderness emergencies
   - Ensure correct symptoms for the condition
   - Treatment options should reflect actual wilderness protocols

Generate a complete, coherent scenario that teaches proper wilderness first aid
through decision-making."""


async def generate_scenario(host_config: HostConfig) -> ScenarioDraft:
    """Generate a complete scenario from minimal host configuration.

    Args:
        host_config: Minimal scenario parameters from the host

    Returns:
        Complete ScenarioDraft with all turns pre-generated

    """
    agent = get_agent(
        agent_name="generator",
        output_type=ScenarioDraft,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        reasoning_effort="high",
    )

    prompt = f"""Generate a wilderness rescue scenario with the following parameters:

Number of Participants: {host_config.num_participants}
Activity Type: {host_config.activity_type}
Difficulty Level: {host_config.difficulty}

Create a complete scenario with:
- Compelling title and setting appropriate for {host_config.activity_type}
- Realistic patient case matching the {host_config.difficulty} difficulty
- 3-5 turns with multiple choice decision points
- Medically accurate content
- Clear learning objectives

The scenario should be challenging but educational for wilderness first responders."""

    result = await agent.run(prompt)
    return result.output
