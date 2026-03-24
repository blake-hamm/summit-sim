"""Simulation agent using PydanticAI for hybrid approach.

In the hybrid model:
- Scenarios are pre-generated with all turns
- Students select from multiple choice options
- AI generates personalized feedback for their choice
"""

from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from summit_sim.agents.config import get_agent
from summit_sim.schemas import (
    ChoiceOption,
    ScenarioDraft,
    ScenarioTurn,
    SimulationResult,
)

FEEDBACK_SYSTEM_PROMPT = """You are a wilderness rescue instructor providing
personalized feedback.

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


SIMULATION_USER_PROMPT = """Scenario: {title}
Setting: {setting}
Patient: {patient_summary}
Hidden Truth: {hidden_truth}
Learning Objectives: {learning_objectives}

Current Situation:
{narrative_text}

Available Choices:
{choices_text}

Student Selected: {selected_choice_id} - {selected_choice_description}

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
    """Process a student's choice and generate personalized feedback.

    In the hybrid model, scenarios have pre-written turns with multiple choice
    options. This agent generates personalized feedback when a student selects
    a choice, then returns the next turn (or marks complete).

    Args:
        scenario: The complete scenario
        current_turn: The current turn the student is on
        selected_choice: The choice option the student selected

    Returns:
        SimulationResult with personalized feedback and next state

    """
    agent = get_agent(
        agent_name="simulation_feedback",
        output_type=SimulationResult,
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
        reasoning_effort="medium",
    )

    choices_text = "\n".join(
        f"- {c.choice_id}: {c.description} (correct={c.is_correct})"
        for c in current_turn.choices
    )

    prompt = SIMULATION_USER_PROMPT.format(
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

    result = await agent.run(prompt)

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
