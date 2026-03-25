"""Tests for the scenario generator agent."""

from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import config as agent_config
from summit_sim.agents.generator import generate_scenario
from summit_sim.schemas import ChoiceOption, ScenarioDraft, ScenarioTurn, TeacherConfig


class TestGeneratorAgent:
    """Tests for the scenario generator agent."""

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
            "Test user prompt with {num_participants} {activity_type} {difficulty}"
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
                "summit_sim.agents.generator.mlflow.genai.load_prompt"
            ) as mock_load_gen,
        ):
            mock_load.return_value = MockPrompt("Test system prompt")
            mock_load_gen.return_value = mock_prompt_obj
            yield

    @pytest.mark.asyncio
    async def test_generate_scenario(self):
        """Test scenario generation from teacher config."""
        teacher_config = TeacherConfig(
            num_participants=4, activity_type="hiking", difficulty="med"
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title="Hiking Emergency",
            setting="Mountain trail at 8,000ft",
            patient_summary="45yo male with chest pain",
            hidden_truth="Possible cardiac event",
            learning_objectives=["Assess chest pain", "Monitor vitals"],
            turns=[
                ScenarioTurn(
                    turn_id=0,
                    narrative_text="Patient complains of chest pain.",
                    choices=[
                        ChoiceOption(
                            choice_id="assess",
                            description="Assess ABCs",
                            is_correct=True,
                            next_turn_id=1,
                        ),
                        ChoiceOption(
                            choice_id="ignore",
                            description="Tell them to rest",
                            is_correct=False,
                            next_turn_id=1,
                        ),
                    ],
                    is_starting_turn=True,
                )
            ],
            starting_turn_id=0,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert result.title == "Hiking Emergency"
        assert len(result.turns) >= 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("activity", ["canyoneering", "skiing", "hiking"])
    async def test_generate_scenario_different_activities(
        self, activity: Literal["canyoneering", "skiing", "hiking"]
    ):
        """Test scenario generation for different activity types."""
        teacher_config = TeacherConfig(
            num_participants=3, activity_type=activity, difficulty="low"
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title=f"{activity.title()} Emergency",
            setting="Test setting",
            patient_summary="Test patient",
            hidden_truth="Test truth",
            learning_objectives=["Objective"],
            turns=[
                ScenarioTurn(
                    turn_id=0,
                    narrative_text="Test scenario.",
                    choices=[
                        ChoiceOption(
                            choice_id="choice_1",
                            description="Option 1",
                            is_correct=True,
                            next_turn_id=None,
                        ),
                        ChoiceOption(
                            choice_id="choice_2",
                            description="Option 2",
                            is_correct=False,
                            next_turn_id=None,
                        ),
                    ],
                    is_starting_turn=True,
                )
            ],
            starting_turn_id=0,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert activity in result.title.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("difficulty", ["low", "med", "high"])
    async def test_generate_scenario_different_difficulties(
        self, difficulty: Literal["low", "med", "high"]
    ):
        """Test scenario generation for different difficulty levels."""
        teacher_config = TeacherConfig(
            num_participants=5, activity_type="hiking", difficulty=difficulty
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title=f"{difficulty.title()} Difficulty Scenario",
            setting="Test setting",
            patient_summary="Test patient",
            hidden_truth="Test truth",
            learning_objectives=["Objective"],
            turns=[
                ScenarioTurn(
                    turn_id=0,
                    narrative_text="Test scenario.",
                    choices=[
                        ChoiceOption(
                            choice_id="choice_1",
                            description="Option 1",
                            is_correct=True,
                            next_turn_id=None,
                        ),
                        ChoiceOption(
                            choice_id="choice_2",
                            description="Option 2",
                            is_correct=False,
                            next_turn_id=None,
                        ),
                    ],
                    is_starting_turn=True,
                )
            ],
            starting_turn_id=0,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert difficulty in result.title.lower()
