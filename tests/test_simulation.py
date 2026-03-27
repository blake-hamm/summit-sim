"""Tests for the simulation feedback agent."""

from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import config as agent_config
from summit_sim.agents.simulation import process_choice
from summit_sim.schemas import (
    ChoiceOption,
    ScenarioDraft,
    ScenarioTurn,
    SimulationResult,
)


class TestSimulationAgent:
    """Tests for the simulation/feedback agent."""

    @pytest.fixture(autouse=True)
    def mock_api_key(self):
        """Mock the API key to avoid errors during agent creation."""
        with patch(
            "summit_sim.agents.config.settings.openrouter_api_key", "test-api-key"
        ):
            yield

    @pytest.fixture(autouse=True)
    def clear_agent_cache(self):
        """Clear the agent cache before each test."""
        agent_config._agent_container.clear()

    @pytest.fixture(autouse=True)
    def mock_prompts(self):
        """Mock MLflow prompt loading."""
        user_prompt = (
            "Test user prompt with {title} {setting} {patient_summary} "
            "{hidden_truth} {learning_objectives} {narrative_text} "
            "{choices_text} {selected_choice_id} {selected_choice_description}"
        )

        class MockPrompt:
            def __init__(self, template):
                self.template = template

            def format(self, **kwargs):
                return self.template.format(**kwargs)

        mock_prompt_obj = MockPrompt(user_prompt)

        with (
            patch("summit_sim.agents.config.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.config.mlflow.genai.register_prompt"),
            patch(
                "summit_sim.agents.simulation.mlflow.genai.load_prompt"
            ) as mock_load_sim,
        ):
            mock_load.return_value = MockPrompt("Test system prompt")
            mock_load_sim.return_value = mock_prompt_obj
            yield

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        turn1 = ScenarioTurn(
            turn_id=0,
            narrative_text="Patient is bleeding.",
            choices=[
                ChoiceOption(
                    choice_id="treat",
                    description="Treat patient",
                    is_correct=True,
                    next_turn_id=1,
                ),
                ChoiceOption(
                    choice_id="wait",
                    description="Wait",
                    is_correct=False,
                    next_turn_id=1,
                ),
                ChoiceOption(
                    choice_id="panic",
                    description="Panic",
                    is_correct=False,
                    next_turn_id=1,
                ),
                ChoiceOption(
                    choice_id="panic",
                    description="Panic",
                    is_correct=False,
                    next_turn_id=1,
                ),
            ],
        )

        turn2 = ScenarioTurn(
            turn_id=1,
            narrative_text="Treatment complete.",
            choices=[
                ChoiceOption(
                    choice_id="monitor",
                    description="Monitor",
                    is_correct=True,
                    next_turn_id=2,
                ),
                ChoiceOption(
                    choice_id="evac",
                    description="Evacuate",
                    is_correct=True,
                    next_turn_id=2,
                ),
                ChoiceOption(
                    choice_id="panic",
                    description="Panic",
                    is_correct=False,
                    next_turn_id=2,
                ),
            ],
        )

        turn3 = ScenarioTurn(
            turn_id=2,
            narrative_text="Patient ready for transport.",
            choices=[
                ChoiceOption(
                    choice_id="package",
                    description="Package for transport",
                    is_correct=True,
                    next_turn_id=None,
                ),
                ChoiceOption(
                    choice_id="wait",
                    description="Wait",
                    is_correct=False,
                    next_turn_id=None,
                ),
                ChoiceOption(
                    choice_id="panic",
                    description="Panic",
                    is_correct=False,
                    next_turn_id=None,
                ),
            ],
        )

        return ScenarioDraft(
            title="Test",
            setting="Location",
            patient_summary="Patient",
            hidden_truth="Truth",
            learning_objectives=["Objective"],
            turns=[turn1, turn2, turn3],
        )

    @pytest.mark.asyncio
    async def test_process_choice_with_next_turn(self, sample_scenario):
        """Test processing a choice that leads to another turn."""
        current_turn = sample_scenario.get_turn(1)
        selected_choice = current_turn.choices[0]

        mock_result = AsyncMock()
        mock_result.output = SimulationResult(
            selected_choice=selected_choice,
            feedback="Good choice!",
            learning_moments=["Learning point"],
            next_turn=None,
            is_complete=False,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await process_choice(
                sample_scenario, current_turn, selected_choice
            )

        assert isinstance(result, SimulationResult)
        assert result.feedback == "Good choice!"
        assert result.is_complete is False
        assert result.next_turn is not None
        assert result.next_turn.turn_id == 2

    @pytest.mark.asyncio
    async def test_process_choice_ends_scenario(self, sample_scenario):
        """Test processing a choice that ends the scenario."""
        current_turn = sample_scenario.get_turn(2)
        selected_choice = current_turn.choices[0]

        mock_result = AsyncMock()
        mock_result.output = SimulationResult(
            selected_choice=selected_choice,
            feedback="Scenario complete!",
            learning_moments=["Final learning"],
            next_turn=None,
            is_complete=True,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await process_choice(
                sample_scenario, current_turn, selected_choice
            )

        assert isinstance(result, SimulationResult)
        assert result.is_complete is True
        assert result.next_turn is None

    @pytest.mark.asyncio
    async def test_process_choice_incorrect_selection(self, sample_scenario):
        """Test processing an incorrect choice selection."""
        current_turn = sample_scenario.get_turn(1)
        selected_choice = current_turn.choices[1]  # The incorrect choice

        mock_result = AsyncMock()
        mock_result.output = SimulationResult(
            selected_choice=selected_choice,
            feedback="Consider applying immediate treatment instead of waiting.",
            learning_moments=["Immediate action is critical in bleeding scenarios"],
            next_turn=None,
            is_complete=False,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await process_choice(
                sample_scenario, current_turn, selected_choice
            )

        assert isinstance(result, SimulationResult)
        assert (
            "consider" in result.feedback.lower()
            or "instead" in result.feedback.lower()
        )
        assert result.is_complete is False
        assert result.next_turn is not None

    @pytest.mark.asyncio
    async def test_process_choice_learning_moments(self, sample_scenario):
        """Test that learning moments are captured."""
        current_turn = sample_scenario.get_turn(1)
        selected_choice = current_turn.choices[0]

        mock_result = AsyncMock()
        mock_result.output = SimulationResult(
            selected_choice=selected_choice,
            feedback="Excellent decision!",
            learning_moments=[
                "Quick assessment is crucial",
                "Always prioritize patient safety",
            ],
            next_turn=None,
            is_complete=False,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await process_choice(
                sample_scenario, current_turn, selected_choice
            )

        assert isinstance(result, SimulationResult)
        assert len(result.learning_moments) == 2
        assert "Quick assessment is crucial" in result.learning_moments
