"""Simulation agent using PydanticAI for hybrid approach.

In the hybrid model:
- Scenarios are pre-generated with all turns
- Students select from multiple choice options
- AI generates personalized feedback for their choice
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.entities import SpanType

from summit_sim.agents.utils import setup_agent_and_prompts
from summit_sim.schemas import (
    ChoiceOption,
    ScenarioDraft,
    ScenarioTurn,
    SimulationResult,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "simulation-feedback"

SYSTEM_PROMPT = """\
You are a wilderness rescue instructor providing personalized feedback.

Your task is to generate educational feedback when a student makes a choice
in a wilderness rescue scenario.

You will receive:
1. The current scenario context
2. The turn they are on
3. The choice they selected
4. The available options they had

You must provide:
- Personalized feedback on their choice (encouraging but honest)
- 1-2 learning moments specific to this decision
- Whether this was the optimal choice or not

Guidelines:
- Be supportive and educational, not judgmental
- Explain WHY the choice was good/suboptimal
- Connect feedback to real wilderness first aid principles
- Keep feedback concise (2-3 sentences)
- Learning moments should be actionable takeaways"""


USER_PROMPT_TEMPLATE = """\
Scenario: {{title}}
Setting: {{setting}}
Patient: {{patient_summary}}
Hidden Truth: {{hidden_truth}}
Learning Objectives: {{learning_objectives}}

Current Situation:
{{narrative_text}}

Available Choices:
{{choices_text}}

Student Selected: {{selected_choice_id}} - {{selected_choice_description}}

Generate personalized feedback for this choice and indicate if the scenario continues.

For the next_turn field:
- If selected_choice.next_turn_id exists, use scenario.get_turn() to fetch it
- If next_turn_id is null, set next_turn=null and is_complete=true
- selected_choice should reference the ChoiceOption passed in"""


@mlflow.trace(span_type=SpanType.AGENT)
async def process_choice(
    scenario: ScenarioDraft,
    current_turn: ScenarioTurn,
    selected_choice: ChoiceOption,
) -> SimulationResult:
    """Process a student's choice and generate personalized feedback."""
    logger.info(
        "Processing choice: turn_id=%s, choice_id=%s",
        current_turn.turn_id,
        selected_choice.choice_id,
    )
    agent, user_prompt = setup_agent_and_prompts(
        agent_name=AGENT_NAME,
        output_type=SimulationResult,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        reasoning_effort="medium",
    )

    choices_text = "\n".join(
        f"- {c.choice_id}: {c.description} (correct={c.is_correct})"
        for c in current_turn.choices
    )

    prompt = user_prompt.format(
        title=scenario.title,
        setting=scenario.setting,
        patient_summary=scenario.patient_summary,
        hidden_truth=scenario.hidden_truth,
        learning_objectives=", ".join(scenario.learning_objectives),
        narrative_text=current_turn.narrative_text,
        choices_text=choices_text,
        selected_choice_id=selected_choice.choice_id,
        selected_choice_description=selected_choice.description,
    )

    result = await agent.run(prompt)  # type: ignore[arg-type]

    simulation_result = result.output

    simulation_result.selected_choice = selected_choice

    if selected_choice.next_turn_id:
        next_turn = scenario.get_turn(selected_choice.next_turn_id)
        simulation_result.next_turn = next_turn
        simulation_result.is_complete = False
    else:
        simulation_result.next_turn = None
        simulation_result.is_complete = True

    return simulation_result
