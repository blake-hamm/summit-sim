"""Tests for the scenario generator agent."""

from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import utils as agent_utils
from summit_sim.agents.generator import (
    AGENT_NAME,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    generate_scenario,
)
from summit_sim.schemas import ScenarioConfig, ScenarioDraft


class TestAgentConstants:
    """Tests for agent constants."""

    def test_agent_name(self):
        """Test that AGENT_NAME is set correctly."""
        assert AGENT_NAME == "generator-draft"

    def test_system_prompt_contains_key_elements(self):
        """Test that system prompt contains required elements."""
        assert "wilderness rescue" in SYSTEM_PROMPT.lower()
        assert (
            "initial_narrative" in SYSTEM_PROMPT.lower()
            or "initial narrative" in SYSTEM_PROMPT.lower()
        )
        assert "hidden_state" in SYSTEM_PROMPT
        assert "scene_state" in SYSTEM_PROMPT

    def test_user_prompt_template_contains_placeholders(self):
        """Test that user prompt template has all required placeholders."""
        assert "{{primary_focus}}" in USER_PROMPT_TEMPLATE
        assert "{{environment}}" in USER_PROMPT_TEMPLATE
        assert "{{available_personnel}}" in USER_PROMPT_TEMPLATE
        assert "{{evac_distance}}" in USER_PROMPT_TEMPLATE
        assert "{{complexity}}" in USER_PROMPT_TEMPLATE


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
            hidden_state="Patient is a 45-year-old male with possible cardiac event. "
            "Blood pressure 140/90, heart rate 110. Pain level 8/10.",
            scene_state="Mountain trail at 8000ft elevation. Weather clear. "
            "No cell coverage available. Temperature 65F.",
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
        assert isinstance(result.hidden_state, str)
        assert isinstance(result.scene_state, str)

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
            hidden_state="Test hidden state",
            scene_state="Test scene state",
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
            hidden_state="Test hidden",
            scene_state="Test scene",
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
        """Test that generated scenario includes state fields as strings."""
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
            hidden_state="Patient has fractured tibia with 7/10 pain. "
            "30 minutes since injury occurred. No medications given.",
            scene_state="Partly cloudy weather, 55F. Sunset in 3 hours. "
            "Spotty cell coverage. Nearest help 4 hours away.",
        )

        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent
            result = await generate_scenario(teacher_config)

        assert isinstance(result, ScenarioDraft)
        assert isinstance(result.hidden_state, str)
        assert isinstance(result.scene_state, str)
        assert "fractured tibia" in result.hidden_state
        assert "cloudy" in result.scene_state.lower()

    @pytest.mark.asyncio
    async def test_generate_scenario_prompt_formatting(self):
        """Test that prompt is formatted with config values."""
        teacher_config = ScenarioConfig(
            primary_focus="Medical",
            environment="Desert",
            available_personnel="Solo Rescuer (1)",
            evac_distance="Remote (1 day)",
            complexity="Critical",
        )

        mock_result = AsyncMock()
        mock_result.output = ScenarioDraft(
            title="Formatted Prompt Test",
            setting="Desert Canyon",
            patient_summary="Solo hiker",
            hidden_truth="Heat stroke",
            learning_objectives=["Recognize heat illness"],
            initial_narrative="A solo hiker collapses in the desert heat.",
            hidden_state="Severe dehydration and heat stroke",
            scene_state="Desert canyon, extreme heat, no shade",
        )

        with patch("summit_sim.agents.generator.setup_agent_and_prompts") as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result

            class MockPrompt:
                def __init__(self, template):
                    self.template = template

                def format(self, **kwargs):
                    return self.template.format(**kwargs)

            mock_user_prompt = MockPrompt(USER_PROMPT_TEMPLATE)
            mock_setup.return_value = (mock_agent, mock_user_prompt)

            await generate_scenario(teacher_config)

            # Verify setup was called with high reasoning effort
            mock_setup.assert_called_once()
            call_kwargs = mock_setup.call_args.kwargs
            assert call_kwargs["reasoning_effort"] == "high"
            assert call_kwargs["agent_name"] == "generator-draft"

    @pytest.mark.asyncio
    async def test_generate_scenario_all_config_combinations(self):
        """Test scenario generation with various config combinations."""
        configs = [
            ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Partner (2)",
                evac_distance="Short (< 2 hours)",
                complexity="Standard",
            ),
            ScenarioConfig(
                primary_focus="Environmental",
                environment="Forest/Trail",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Complicated",
            ),
            ScenarioConfig(
                primary_focus="Mixed",
                environment="Desert",
                available_personnel="Large Expedition (6+)",
                evac_distance="Expedition (2+ days)",
                complexity="Critical",
            ),
        ]

        for config in configs:
            # Clear agent cache for each config iteration
            agent_utils._agent_container.clear()

            mock_result = AsyncMock()
            mock_result.output = ScenarioDraft(
                title=f"Test {config.primary_focus}",
                setting=config.environment,
                patient_summary="Test patient",
                hidden_truth="Test condition",
                learning_objectives=["Learn"],
                initial_narrative="Test narrative",
                hidden_state="Test hidden state",
                scene_state="Test scene state",
            )

            with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run.return_value = mock_result
                mock_agent_class.return_value = mock_agent

                result = await generate_scenario(config)

                assert isinstance(result, ScenarioDraft)
                assert result.setting == config.environment
