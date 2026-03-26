"""Tests for agent configuration."""

from unittest.mock import MagicMock, patch

from summit_sim.agents.config import _get_or_register_system_prompt, get_agent


def test_get_or_register_system_prompt_existing():
    """Test getting existing system prompt from MLflow."""
    with patch("summit_sim.agents.config.mlflow") as mock_mlflow:
        # Mock the load_prompt to return a prompt object
        mock_prompt = MagicMock()
        mock_prompt.template = "Existing system prompt"
        mock_mlflow.genai.load_prompt.return_value = mock_prompt

        prompt = _get_or_register_system_prompt("test-agent", "New system prompt")

        # Should return the existing prompt, not register a new one
        assert prompt == "Existing system prompt"
        mock_mlflow.genai.load_prompt.assert_called_once()
        mock_mlflow.genai.register_prompt.assert_not_called()


def test_get_or_register_system_prompt_new():
    """Test registering new system prompt when not found."""
    with patch("summit_sim.agents.config.mlflow") as mock_mlflow:
        # Mock the load_prompt to raise an exception (prompt not found)
        mock_mlflow.genai.load_prompt.side_effect = Exception("Prompt not found")

        prompt = _get_or_register_system_prompt("test-agent", "New system prompt")

        # Should register the new prompt and return it
        assert prompt == "New system prompt"
        mock_mlflow.genai.load_prompt.assert_called_once()
        mock_mlflow.genai.register_prompt.assert_called_once_with(
            name="test-agent-system",
            template="New system prompt",
        )


def test_get_agent_creates_new():
    """Test that get_agent creates a new agent when not cached."""
    with (
        patch("summit_sim.agents.config.OpenRouterProvider"),
        patch("summit_sim.agents.config.OpenRouterModel"),
        patch("summit_sim.agents.config.OpenRouterModelSettings"),
        patch(
            "summit_sim.agents.config._get_or_register_system_prompt"
        ) as mock_get_prompt,
        patch("summit_sim.agents.config.Agent") as mock_agent_class,
    ):
        mock_get_prompt.return_value = "Test system prompt"
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # Import inside test to avoid caching issues
        from summit_sim.agents.config import _agent_container  # noqa: PLC0415

        # Clear the container to ensure we're testing creation
        if "test-agent" in _agent_container:
            del _agent_container["test-agent"]

        agent = get_agent(
            agent_name="test-agent",
            output_type=str,
            system_prompt="Test system prompt",
            reasoning_effort="medium",
        )

        # Should have created a new agent
        mock_get_prompt.assert_called_once_with("test-agent", "Test system prompt")
        mock_agent_class.assert_called_once()
        assert agent == mock_agent_instance
        # Should be cached now
        assert "test-agent" in _agent_container


def test_get_agent_returns_cached():
    """Test that get_agent returns cached agent when available."""
    from summit_sim.agents.config import _agent_container  # noqa: PLC0415

    # Pre-populate the cache
    mock_agent = MagicMock()
    _agent_container["test-agent"] = mock_agent

    try:
        agent = get_agent(
            agent_name="test-agent",
            output_type=str,
            system_prompt="Test system prompt",
            reasoning_effort="medium",
        )

        # Should return the cached agent
        assert agent == mock_agent
    finally:
        # Clean up
        if "test-agent" in _agent_container:
            del _agent_container["test-agent"]
