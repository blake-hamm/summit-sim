"""Tests for the scenario image generator agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
        """Test successful image generation with base64 response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "images": [
                            {
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        "/9j/4AAQSkZJRgABAQAAAQABAA"
                                    )
                                }
                            }
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
            patch(
                "summit_sim.agents.image_generator.settings.image_generation_model",
                "test-model",
            ),
            patch(
                "summit_sim.agents.image_generator.settings.image_generation_timeout",
                60,
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result == "/9j/4AAQSkZJRgABAQAAAQABAA"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_image_uses_custom_model(
        self, sample_scenario, sample_config
    ):
        """Test that custom model parameter is respected."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "data:image/jpeg;base64,test123"}}
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        custom_model = "custom-image-model"

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
        ):
            result = await generate_scenario_image(
                sample_scenario, sample_config, model=custom_model
            )

        # Verify the model was passed in the request
        call_args = mock_client.post.call_args
        json_payload = call_args.kwargs["json"]
        assert json_payload["model"] == custom_model
        assert result == "test123"

    @pytest.mark.asyncio
    async def test_generate_image_no_choices_in_response(
        self, sample_scenario, sample_config
    ):
        """Test handling of response with no choices."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_no_images_in_message(
        self, sample_scenario, sample_config
    ):
        """Test handling of response with no images in message."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"images": []}}]}
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_unexpected_url_format(
        self, sample_scenario, sample_config
    ):
        """Test handling of unexpected image URL format (not base64)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "https://example.com/image.jpg"}}
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_timeout(self, sample_scenario, sample_config):
        """Test handling of HTTP timeout."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
            patch(
                "summit_sim.agents.image_generator.settings.image_generation_timeout",
                30,
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_http_error(self, sample_scenario, sample_config):
        """Test handling of HTTP error response."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "500 Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_general_exception(
        self, sample_scenario, sample_config
    ):
        """Test handling of general exceptions."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Unexpected error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
        ):
            result = await generate_scenario_image(sample_scenario, sample_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_image_request_structure(
        self, sample_scenario, sample_config
    ):
        """Test that the HTTP request has correct structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "images": [
                            {"image_url": {"url": "data:image/jpeg;base64,testdata"}}
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "summit_sim.agents.image_generator.settings.openrouter_api_key",
                "test-api-key",
            ),
            patch(
                "summit_sim.agents.image_generator.settings.image_generation_model",
                "openai/gpt-image-1",
            ),
        ):
            await generate_scenario_image(sample_scenario, sample_config)

        # Verify request structure
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://openrouter.ai/api/v1/chat/completions"
        assert "Authorization" in call_args.kwargs["headers"]
        assert "Bearer test-api-key" in call_args.kwargs["headers"]["Authorization"]

        json_payload = call_args.kwargs["json"]
        assert "model" in json_payload
        assert "messages" in json_payload
        assert json_payload["modalities"] == ["image"]
        assert json_payload["image_config"]["aspect_ratio"] == "16:9"
        assert len(json_payload["messages"]) == 1
        assert json_payload["messages"][0]["role"] == "user"
