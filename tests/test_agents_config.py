"""Tests for agent configuration."""

from unittest.mock import MagicMock, patch

from summit_sim.agents.utils import (
    _agent_container,
    _get_or_register_prompt,
    setup_agent_and_prompts,
)


def test_get_or_register_prompt_existing_unchanged():
    """Test getting existing prompt when content unchanged."""
    with patch("summit_sim.agents.utils.mlflow") as mock_mlflow:
        # Mock the load_prompt to return a prompt object with same content
        mock_prompt = MagicMock()
        mock_prompt.template = "Same prompt"
        mock_mlflow.genai.load_prompt.return_value = mock_prompt

        result = _get_or_register_prompt("test-agent-system", "Same prompt")

        # Should return the loaded prompt object without registering new version
        assert result == mock_prompt
        assert (
            mock_mlflow.genai.load_prompt.call_count == 2
        )  # Once to check, once to return
        mock_mlflow.genai.register_prompt.assert_not_called()


def test_get_or_register_prompt_existing_changed():
    """Test registering new version when prompt content changed."""
    with patch("summit_sim.agents.utils.mlflow") as mock_mlflow:
        # Mock the load_prompt to return a prompt object with different content
        mock_prompt = MagicMock()
        mock_prompt.template = "Old prompt"
        mock_mlflow.genai.load_prompt.return_value = mock_prompt

        result = _get_or_register_prompt("test-agent-system", "New prompt")

        # Should register new version and return loaded prompt
        assert result == mock_prompt
        assert mock_mlflow.genai.load_prompt.call_count == 2
        mock_mlflow.genai.register_prompt.assert_called_once_with(
            name="test-agent-system",
            template="New prompt",
        )


def test_get_or_register_prompt_new():
    """Test registering new prompt when not found."""
    with patch("summit_sim.agents.utils.mlflow") as mock_mlflow:
        # Mock the load_prompt to raise an exception (prompt not found)
        mock_prompt = MagicMock()
        mock_mlflow.genai.load_prompt.side_effect = [
            Exception("Prompt not found"),
            mock_prompt,  # Return mock on second call after registration
        ]

        result = _get_or_register_prompt("test-agent-system", "New prompt")

        # Should register the new prompt and return loaded prompt
        assert result == mock_prompt
        mock_mlflow.genai.register_prompt.assert_called_once_with(
            name="test-agent-system",
            template="New prompt",
        )


def test_setup_agent_and_prompts_creates_new():
    """Test that setup_agent_and_prompts creates a new agent when not cached."""
    with (
        patch("summit_sim.agents.utils.OpenRouterProvider"),
        patch("summit_sim.agents.utils.OpenRouterModel"),
        patch("summit_sim.agents.utils.OpenRouterModelSettings"),
        patch("summit_sim.agents.utils._get_or_register_prompt") as mock_get_prompt,
        patch("summit_sim.agents.utils.Agent") as mock_agent_class,
    ):
        # Mock system prompt object with template
        mock_system_obj = MagicMock()
        mock_system_obj.template = "Test system prompt"
        mock_user_obj = MagicMock()
        mock_user_obj.template = "Test user prompt {{var}}"

        # Return system obj first (for agent creation), then user obj
        mock_get_prompt.side_effect = [mock_system_obj, mock_user_obj]

        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # Clear the container to ensure we're testing creation
        if "test-agent" in _agent_container:
            del _agent_container["test-agent"]

        agent, user_prompt = setup_agent_and_prompts(
            agent_name="test-agent",
            output_type=str,
            system_prompt="Test system prompt",
            user_prompt_template="Test user prompt {{var}}",
            reasoning_effort="medium",
        )

        # Should have created a new agent and returned tuple with user prompt object
        assert mock_get_prompt.call_count == 2
        mock_get_prompt.assert_any_call("test-agent-system", "Test system prompt")
        mock_get_prompt.assert_any_call("test-agent-user", "Test user prompt {{var}}")
        mock_agent_class.assert_called_once()

        # Verify agent was created with system prompt template
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["system_prompt"] == "Test system prompt"

        assert agent == mock_agent_instance
        assert user_prompt == mock_user_obj
        # Should be cached now
        assert "test-agent" in _agent_container


def test_setup_agent_and_prompts_returns_cached():
    """Test that setup_agent_and_prompts returns cached agent when available."""
    # Pre-populate the cache
    mock_agent = MagicMock()
    _agent_container["test-agent"] = mock_agent

    try:
        with patch(
            "summit_sim.agents.utils._get_or_register_prompt"
        ) as mock_get_prompt:
            mock_user_obj = MagicMock()
            mock_get_prompt.return_value = mock_user_obj

            agent, user_prompt = setup_agent_and_prompts(
                agent_name="test-agent",
                output_type=str,
                system_prompt="Test system prompt",
                user_prompt_template="Test user prompt {{var}}",
                reasoning_effort="medium",
            )

            # Should return the cached agent and user prompt object
            # Only called once for user prompt
            # (system prompt not needed for cached agent)
            assert agent == mock_agent
            assert user_prompt == mock_user_obj
            mock_get_prompt.assert_called_once_with(
                "test-agent-user", "Test user prompt {{var}}"
            )
    finally:
        # Clean up
        if "test-agent" in _agent_container:
            del _agent_container["test-agent"]
