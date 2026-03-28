"""Tests for the scenario generator agent."""

from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import utils as agent_utils
from summit_sim.agents.generator import generate_scenario
from summit_sim.schemas import ScenarioConfig, ScenarioDraft


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
            initial_narrative=(
                "You arrive at a mountain trail to find a 45-year-old male "
                "clutching his chest. He appears pale and is sweating profusely. "
                "The patient reports chest pain that started 20 minutes ago "
                "while hiking."
            ),
            hidden_state={
                "diagnosis": "possible cardiac event",
                "vitals": "bp 140/90, hr 110",
                "pain_level": "8/10",
            },
            scene_state={
                "elevation": "8000ft",
                "weather": "clear",
                "cell_coverage": "none",
            },
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert result.title == "Hiking Emergency"
        assert result.initial_narrative is not None
        assert len(result.initial_narrative) > 0

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
            initial_narrative=f"A {focus.lower()} emergency scenario begins.",
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert focus.lower() in result.title.lower()
        assert result.initial_narrative is not None

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
            initial_narrative=f"A {complexity.lower()} complexity scenario begins.",
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert complexity.lower() in result.title.lower()
        assert result.initial_narrative is not None

    @pytest.mark.asyncio
    async def test_generate_scenario_with_state(self):
        """Test that generated scenario includes state fields."""
        teacher_config = ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Small Group (3-5)",
            evac_distance="Remote (1 day)",
            complexity="Standard",
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title="State Test Scenario",
            setting="Mountain trail",
            patient_summary="Patient with injury",
            hidden_truth="Fractured tibia",
            learning_objectives=["Assess injury", "Immobilize limb"],
            initial_narrative="You find a hiker with a leg injury on the trail.",
            hidden_state={
                "injury": "fractured tibia",
                "pain_scale": "7/10",
                "time_since_injury": "30 minutes",
            },
            scene_state={
                "weather": "partly cloudy",
                "temperature": "55F",
                "sunset_hours": "3",
                "cell_coverage": "spotty",
            },
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert "injury" in result.hidden_state
        assert "weather" in result.scene_state
