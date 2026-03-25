"""Scenario generator agent using PydanticAI."""

from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from summit_sim.agents.config import get_agent
from summit_sim.schemas import ScenarioDraft, TeacherConfig

AGENT_NAME = "generator-draft"

SYSTEM_PROMPT = """\
You are an expert wilderness rescue scenario designer.

Your task is to create a realistic, medically accurate wilderness rescue scenario
based on minimal teacher inputs. You must generate a complete scenario with ALL
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

5. Turn IDs should be sequential integers starting from 0
    (e.g., 0, 1, 2, 3, 4). Use 0 for the starting turn.

6. Medical accuracy:
   - Base scenarios on common wilderness emergencies
   - Ensure correct symptoms for the condition
   - Treatment options should reflect actual wilderness protocols

Generate a complete, coherent scenario that teaches proper wilderness first aid
through decision-making."""


USER_PROMPT_TEMPLATE = """\
Generate a wilderness rescue scenario with the following parameters:

Number of Participants: {{num_participants}}
Activity Type: {{activity_type}}
Difficulty Level: {{difficulty}}

Create a complete scenario with:
- Compelling title and setting appropriate for {{activity_type}}
- Realistic patient case matching the {{difficulty}} difficulty
- 3-5 turns with multiple choice decision points
- Medically accurate content
- Clear learning objectives

The scenario should be challenging but educational for wilderness first responders."""


@mlflow.trace(span_type=SpanType.AGENT)
async def generate_scenario(teacher_config: TeacherConfig) -> ScenarioDraft:
    """Generate a complete scenario from minimal teacher configuration.

    Args:
        teacher_config: Minimal scenario parameters from the teacher

    Returns:
        Complete ScenarioDraft with all turns pre-generated

    """
    agent = get_agent(
        agent_name=AGENT_NAME,
        output_type=ScenarioDraft,
        system_prompt=SYSTEM_PROMPT,
        reasoning_effort="high",
    )

    user_prompt = mlflow.genai.load_prompt(  # type: ignore[attr-defined]
        f"prompts:/{AGENT_NAME}-user@latest"
    )
    prompt = user_prompt.format(
        num_participants=teacher_config.num_participants,
        activity_type=teacher_config.activity_type,
        difficulty=teacher_config.difficulty,
    )

    result = await agent.run(prompt)
    return result.output
