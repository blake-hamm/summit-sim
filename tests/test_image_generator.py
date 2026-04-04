"""Tests for the scenario image generator agent."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summit_sim.agents.image_generator import (
    IMAGE_PROMPT_TEMPLATE,
    build_image_prompt,
    generate_scenario_image,
)
from summit_sim.schemas import ScenarioConfig, ScenarioDraft


class TestImagePromptConstants:
    """Tests for image prompt constants."""

    def test_prompt_template_contains_placeholders(self):
        """Test that prompt template has all required placeholders."""
        assert "{title}" in IMAGE_PROMPT_TEMPLATE
        assert "{setting}" in IMAGE_PROMPT_TEMPLATE
        assert "{environment}" in IMAGE_PROMPT_TEMPLATE
        assert "{available_personnel}" in IMAGE_PROMPT_TEMPLATE
        assert "{evac_distance}" in IMAGE_PROMPT_TEMPLATE
        assert "{complexity}" in IMAGE_PROMPT_TEMPLATE
        assert "{primary_focus}" in IMAGE_PROMPT_TEMPLATE

    def test_prompt_template_mentions_wilderness(self):
        """Test that prompt mentions wilderness for context."""
        assert "wilderness" in IMAGE_PROMPT_TEMPLATE.lower()


class TestBuildImagePrompt:
    """Tests for build_image_prompt function."""

    def test_build_prompt_with_all_fields(self):
        """Test building prompt with all scenario fields populated."""
        scenario = ScenarioDraft(
            title="Mountain Emergency",
            setting=(
                "Rocky mountain trail at 9,000ft elevation, "
                "late afternoon with scattered clouds"
            ),
            patient_summary="Hiker with leg injury",
            hidden_truth="Fractured femur",
            learning_objectives=["Assess injury", "Splint properly"],
            initial_narrative="You encounter an injured hiker...",
            hidden_state="Fractured femur, BP 120/80",
            scene_state="Cool weather, 50F, wind 10mph",
        )

        config = ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Small Group (3-5)",
            evac_distance="Remote (1 day)",
            complexity="Standard",
        )

        prompt = build_image_prompt(scenario, config)

        assert scenario.title in prompt
        assert scenario.setting in prompt
        assert config.environment in prompt
        assert config.available_personnel in prompt
        assert config.evac_distance in prompt
        assert config.complexity in prompt
        assert config.primary_focus in prompt

    def test_build_prompt_with_different_configs(self):
        """Test building prompt with various config combinations."""
        configs = [
            ScenarioConfig(
                primary_focus="Medical",
                environment="Desert",
                available_personnel="Solo Rescuer (1)",
                evac_distance="Short (< 2 hours)",
                complexity="Critical",
            ),
            ScenarioConfig(
                primary_focus="Environmental",
                environment="Winter/Snow",
                available_personnel="Partner (2)",
                evac_distance="Expedition (2+ days)",
                complexity="Complicated",
            ),
        ]

        scenario = ScenarioDraft(
            title="Test Scenario",
            setting="Test setting description",
            patient_summary="Test patient",
            hidden_truth="Test condition",
            learning_objectives=["Learn", "Practice"],
            initial_narrative="Test narrative",
            hidden_state="Test hidden",
            scene_state="Test scene",
        )

        for config in configs:
            prompt = build_image_prompt(scenario, config)
            assert config.environment in prompt
            assert config.primary_focus in prompt


class TestGenerateScenarioImage:
    """Tests for generate_scenario_image async function."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        return ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail test setting",
            patient_summary="Test patient",
            hidden_truth="Test condition",
            learning_objectives=["Learn", "Practice"],
            initial_narrative="Test narrative",
            hidden_state="Test hidden",
            scene_state="Test scene",
        )

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Partner (2)",
            evac_distance="Short (< 2 hours)",
            complexity="Standard",
        )

    @pytest.mark.asyncio
    async def test_generate_image_success(self, sample_scenario, sample_config):
        """Test successful image generation with Vertex AI response."""
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"test_imagedata"

        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_models = MagicMock()
        mock_models.generate_content = MagicMock(return_value=mock_response)

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        mock_provider = MagicMock()
        mock_provider.client = mock_client

        with patch(
            "summit_sim.agents.image_generator.get_provider",
            return_value=mock_provider,
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result == base64.b64encode(b"test_imagedata").decode("ascii")
        mock_aio_models.generate_content.assert_called_once()
        call_args = mock_aio_models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemini-3.1-flash-image-preview"
        assert call_args.kwargs["config"].response_modalities == ["IMAGE"]

    @pytest.mark.asyncio
    async def test_generate_image_uses_custom_model(
        self, sample_scenario, sample_config
    ):
        """Test that custom model parameter is respected."""
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"testimagedata"

        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        mock_provider = MagicMock()
        mock_provider.client = mock_client

        custom_model = "custom-image-model"

        with patch(
            "summit_sim.agents.image_generator.get_provider",
            return_value=mock_provider,
        ):
            await generate_scenario_image(
                sample_scenario, sample_config, model=custom_model
            )

        call_args = mock_aio_models.generate_content.call_args
        assert call_args.kwargs["model"] == custom_model

    @pytest.mark.asyncio
    async def test_generate_image_no_candidates(self, sample_scenario, sample_config):
        """Test handling of response with no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = None

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        mock_provider = MagicMock()
        mock_provider.client = mock_client

        with patch(
            "summit_sim.agents.image_generator.get_provider",
            return_value=mock_provider,
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_no_content(self, sample_scenario, sample_config):
        """Test handling of response with no content."""
        mock_candidate = MagicMock()
        mock_candidate.content = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        mock_provider = MagicMock()
        mock_provider.client = mock_client

        with patch(
            "summit_sim.agents.image_generator.get_provider",
            return_value=mock_provider,
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_no_inline_data(self, sample_scenario, sample_config):
        """Test handling of response with no inline_data in parts."""
        mock_part = MagicMock()
        mock_part.inline_data = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        mock_provider = MagicMock()
        mock_provider.client = mock_client

        with patch(
            "summit_sim.agents.image_generator.get_provider",
            return_value=mock_provider,
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_exception(self, sample_scenario, sample_config):
        """Test handling of exceptions during image generation."""
        mock_aio_models = MagicMock()
        mock_aio_models.generate_content = MagicMock(side_effect=Exception("API error"))

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models

        mock_provider = MagicMock()
        mock_provider.client = mock_client

        with patch(
            "summit_sim.agents.image_generator.get_provider",
            return_value=mock_provider,
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None
