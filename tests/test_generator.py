"""Tests for the scenario generator agent."""

from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import utils as agent_utils
from summit_sim.agents.generator import generate_scenario
from summit_sim.schemas import ChoiceOption, ScenarioConfig, ScenarioDraft, ScenarioTurn


class TestGeneratorAgent:
    """Tests for the scenario generator agent."""

    @pytest.fixture(autouse=True)
    def mock_api_key(self):
        """Mock the API key to avoid errors during agent creation."""
        with patch(
            "summit_sim.agents.utils.settings.openrouter_api_key", "test-api-key"
        ):
            yield

    @pytest.fixture(autouse=True)
    def clear_agent_cache(self):
        """Clear the agent cache before each test."""
        agent_utils._agent_container.clear()

    @pytest.fixture(autouse=True)
    def mock_prompts(self):
        """Mock MLflow prompt loading."""

        class MockPrompt:
            def __init__(self, template):
                self.template = template

            def format(self, **kwargs):
                return self.template.format(**kwargs)

        with (
            patch("summit_sim.agents.utils.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.utils.mlflow.genai.register_prompt"),
        ):
            mock_load.return_value = MockPrompt("Test system prompt")
            yield

    @pytest.mark.asyncio
    async def test_generate_scenario(self):
        """Test scenario generation from teacher config."""
        teacher_config = ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Small Group (3-5)",
            evac_distance="Remote (1 day)",
            complexity="Standard",
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
                        ChoiceOption(
                            choice_id="panic",
                            description="Panic",
                            is_correct=False,
                            next_turn_id=1,
                        ),
                    ],
                ),
                ScenarioTurn(
                    turn_id=1,
                    narrative_text="Patient is now stable.",
                    choices=[
                        ChoiceOption(
                            choice_id="monitor",
                            description="Monitor vitals",
                            is_correct=True,
                            next_turn_id=2,
                        ),
                        ChoiceOption(
                            choice_id="evac",
                            description="Evacuate immediately",
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
                ),
                ScenarioTurn(
                    turn_id=2,
                    narrative_text="Transport arriving.",
                    choices=[
                        ChoiceOption(
                            choice_id="prepare",
                            description="Prepare for transport",
                            is_correct=True,
                            next_turn_id=None,
                        ),
                        ChoiceOption(
                            choice_id="wait",
                            description="Wait for instructions",
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
                ),
            ],
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert result.title == "Hiking Emergency"
        assert len(result.turns) >= 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focus", ["Trauma", "Medical", "Environmental", "Mixed"])
    async def test_generate_scenario_different_focus(
        self, focus: Literal["Trauma", "Medical", "Environmental", "Mixed"]
    ):
        """Test scenario generation for different WFR curriculum focus."""
        teacher_config = ScenarioConfig(
            primary_focus=focus,
            environment="Forest/Trail",
            available_personnel="Partner (2)",
            evac_distance="Short (< 2 hours)",
            complexity="Standard",
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title=f"{focus} Emergency",
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
                            next_turn_id=1,
                        ),
                        ChoiceOption(
                            choice_id="choice_2",
                            description="Option 2",
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
                ),
                ScenarioTurn(
                    turn_id=1,
                    narrative_text="Next turn.",
                    choices=[
                        ChoiceOption(
                            choice_id="continue",
                            description="Continue",
                            is_correct=True,
                            next_turn_id=2,
                        ),
                        ChoiceOption(
                            choice_id="stop",
                            description="Stop",
                            is_correct=False,
                            next_turn_id=2,
                        ),
                        ChoiceOption(
                            choice_id="panic",
                            description="Panic",
                            is_correct=False,
                            next_turn_id=2,
                        ),
                    ],
                ),
                ScenarioTurn(
                    turn_id=2,
                    narrative_text="Final turn.",
                    choices=[
                        ChoiceOption(
                            choice_id="finish",
                            description="Finish",
                            is_correct=True,
                            next_turn_id=None,
                        ),
                        ChoiceOption(
                            choice_id="continue",
                            description="Continue",
                            is_correct=True,
                            next_turn_id=None,
                        ),
                        ChoiceOption(
                            choice_id="panic",
                            description="Panic",
                            is_correct=False,
                            next_turn_id=None,
                        ),
                    ],
                ),
            ],
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert focus.lower() in result.title.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("complexity", ["Standard", "Complicated", "Critical"])
    async def test_generate_scenario_different_complexity(
        self, complexity: Literal["Standard", "Complicated", "Critical"]
    ):
        """Test scenario generation for different complexity levels."""
        teacher_config = ScenarioConfig(
            primary_focus="Medical",
            environment="Winter/Snow",
            available_personnel="Large Expedition (6+)",
            evac_distance="Expedition (2+ days)",
            complexity=complexity,
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title=f"{complexity} Complexity Scenario",
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
                            next_turn_id=1,
                        ),
                        ChoiceOption(
                            choice_id="choice_2",
                            description="Option 2",
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
                ),
                ScenarioTurn(
                    turn_id=1,
                    narrative_text="Next turn.",
                    choices=[
                        ChoiceOption(
                            choice_id="continue",
                            description="Continue",
                            is_correct=True,
                            next_turn_id=2,
                        ),
                        ChoiceOption(
                            choice_id="stop",
                            description="Stop",
                            is_correct=False,
                            next_turn_id=2,
                        ),
                        ChoiceOption(
                            choice_id="panic",
                            description="Panic",
                            is_correct=False,
                            next_turn_id=2,
                        ),
                    ],
                ),
                ScenarioTurn(
                    turn_id=2,
                    narrative_text="Final turn.",
                    choices=[
                        ChoiceOption(
                            choice_id="finish",
                            description="Finish",
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
                ),
            ],
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert complexity.lower() in result.title.lower()
