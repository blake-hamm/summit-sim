"""Tests for agent utilities module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from summit_sim.agents import utils as agent_utils
from summit_sim.agents.action_responder import ActionResponse
from summit_sim.schemas import ScenarioDraft


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_provider_singleton(self):
        """Test that get_provider returns a singleton instance."""
        provider1 = agent_utils.get_provider()
        provider2 = agent_utils.get_provider()
        assert provider1 is provider2

    def test_get_provider_uses_credentials(self):
        """Test that provider is configured with GCP credentials."""
        agent_utils.get_provider.cache_clear()

        with (
            patch(
                "summit_sim.agents.utils.settings.google_application_credentials",
                Path("/tmp/test-sa.json"),
            ),
            patch("summit_sim.agents.utils.settings.gcp_project_id", "test-project"),
            patch("summit_sim.agents.utils.settings.gcp_location", "us-central1"),
            patch("summit_sim.agents.utils.service_account.Credentials") as mock_creds,
            patch("summit_sim.agents.utils.GoogleProvider") as mock_provider_class,
        ):
            mock_creds.from_service_account_file.return_value = MagicMock()
            mock_provider = MagicMock()
            mock_provider_class.return_value = mock_provider

            provider = agent_utils.get_provider()

            mock_creds.from_service_account_file.assert_called_once()
            mock_provider_class.assert_called_once()
            assert provider == mock_provider

        agent_utils.get_provider.cache_clear()


class TestGetOrRegisterPrompt:
    """Tests for _get_or_register_prompt function."""

    def test_get_existing_prompt_unchanged(self):
        """Test loading an existing prompt that hasn't changed."""
        mock_prompt = MagicMock()
        mock_prompt.template = "unchanged template"

        with (
            patch("summit_sim.agents.utils.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.utils.mlflow.genai.register_prompt") as mock_reg,
        ):
            mock_load.return_value = mock_prompt

            result = agent_utils._get_or_register_prompt(
                "test-prompt", "unchanged template"
            )

            assert result == mock_prompt
            mock_reg.assert_not_called()

    def test_register_new_prompt_when_changed(self):
        """Test registering new version when template changes."""
        mock_prompt = MagicMock()
        mock_prompt.template = "old template"
        new_prompt = MagicMock()
        new_prompt.template = "new template"

        with (
            patch("summit_sim.agents.utils.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.utils.mlflow.genai.register_prompt") as mock_reg,
        ):
            mock_load.side_effect = [mock_prompt, new_prompt]
            mock_reg.return_value = None

            result = agent_utils._get_or_register_prompt("test-prompt", "new template")

            mock_reg.assert_called_once_with(
                name="test-prompt",
                template="new template",
            )
            assert result == new_prompt

    def test_register_new_prompt_when_not_exists(self):
        """Test registering prompt when it doesn't exist yet."""
        new_prompt = MagicMock()
        new_prompt.template = "new template"

        with (
            patch("summit_sim.agents.utils.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.utils.mlflow.genai.register_prompt") as mock_reg,
        ):
            mock_load.side_effect = [Exception("Prompt not found"), new_prompt]
            mock_reg.return_value = None

            result = agent_utils._get_or_register_prompt("test-prompt", "new template")

            mock_reg.assert_called_once_with(
                name="test-prompt",
                template="new template",
            )
            assert result == new_prompt


class TestSetupAgentAndPrompts:
    """Tests for setup_agent_and_prompts function."""

    def test_creates_new_agent(self):
        """Test that function creates a new agent when not cached."""
        mock_prompt = MagicMock()
        mock_prompt.template = "test template"

        with (
            patch("summit_sim.agents.utils._get_or_register_prompt") as mock_get_prompt,
            patch("summit_sim.agents.utils.Agent") as mock_agent_class,
            patch("summit_sim.agents.utils.get_provider"),
        ):
            mock_get_prompt.side_effect = [mock_prompt, mock_prompt]
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent, user_prompt = agent_utils.setup_agent_and_prompts(
                agent_name="test-agent",
                output_type=ScenarioDraft,
                system_prompt="System prompt",
                user_prompt_template="User {variable}",
            )

            assert agent == mock_agent
            assert user_prompt == mock_prompt
            mock_agent_class.assert_called_once()

    def test_returns_cached_agent(self):
        """Test that function returns cached agent on subsequent calls."""
        mock_prompt = MagicMock()
        mock_prompt.template = "test template"
        cached_agent = MagicMock()
        cached_user_prompt = MagicMock()

        agent_utils._agent_container["cached-agent"] = (
            cached_agent,
            cached_user_prompt,
        )

        with patch(
            "summit_sim.agents.utils._get_or_register_prompt"
        ) as mock_get_prompt:
            mock_get_prompt.return_value = mock_prompt

            agent, user_prompt = agent_utils.setup_agent_and_prompts(
                agent_name="cached-agent",
                output_type=ScenarioDraft,
                system_prompt="System prompt",
                user_prompt_template="User {variable}",
            )

            assert agent == cached_agent
            assert user_prompt == cached_user_prompt
            mock_get_prompt.assert_not_called()

    def test_agent_configuration(self):
        """Test that agent is configured with correct parameters."""
        mock_prompt = MagicMock()
        mock_prompt.template = "system prompt content"
        mock_provider = MagicMock()

        with (
            patch("summit_sim.agents.utils._get_or_register_prompt") as mock_get_prompt,
            patch("summit_sim.agents.utils.Agent") as mock_agent_class,
            patch("summit_sim.agents.utils.get_provider", return_value=mock_provider),
            patch("summit_sim.agents.utils.GoogleModel"),
        ):
            mock_get_prompt.side_effect = [mock_prompt, mock_prompt]
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            agent_utils.setup_agent_and_prompts(
                agent_name="config-test",
                output_type=ActionResponse,
                system_prompt="System prompt",
                user_prompt_template="User {var}",
                reasoning_effort="high",
            )

            call_kwargs = mock_agent_class.call_args.kwargs
            assert call_kwargs["output_type"] == ActionResponse
            assert call_kwargs["system_prompt"] == "system prompt content"

    def test_different_reasoning_efforts(self):
        """Test agent creation with different reasoning effort levels."""
        mock_prompt = MagicMock()
        mock_prompt.template = "prompt"

        for effort in ["low", "medium", "high"]:
            agent_utils._agent_container.clear()

            with (
                patch(
                    "summit_sim.agents.utils._get_or_register_prompt"
                ) as mock_get_prompt,
                patch("summit_sim.agents.utils.Agent") as mock_agent_class,
                patch("summit_sim.agents.utils.get_provider"),
                patch("summit_sim.agents.utils.GoogleModel"),
            ):
                mock_get_prompt.side_effect = [mock_prompt, mock_prompt]
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent

                agent_utils.setup_agent_and_prompts(
                    agent_name=f"test-{effort}",
                    output_type=ScenarioDraft,
                    system_prompt="System",
                    user_prompt_template="User",
                    reasoning_effort=effort,  # type: ignore[arg-type]
                )

                mock_agent_class.assert_called_once()


class TestAgentContainer:
    """Tests for the agent container cache."""

    def test_container_is_dict(self):
        """Test that _agent_container is a dictionary."""
        assert isinstance(agent_utils._agent_container, dict)

    def test_container_persists_between_calls(self):
        """Test that container persists agents between calls."""
        mock_agent = MagicMock()
        agent_utils._agent_container["test-agent"] = mock_agent

        assert agent_utils._agent_container["test-agent"] == mock_agent

        del agent_utils._agent_container["test-agent"]

    @pytest.mark.usefixtures("_clear_agent_cache")
    def test_clear_agent_cache_fixture(self):
        """Test that clear_agent_cache fixture works."""
        agent_utils._agent_container["temp-agent"] = MagicMock()
