"""Pytest configuration and shared fixtures for summit-sim tests."""

from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import utils as agent_utils


class MockPrompt:
    """Mock MLflow prompt object for testing."""

    def __init__(self, template: str) -> None:
        """Initialize mock prompt with template."""
        self.template = template

    def format(self, **kwargs) -> str:
        """Format the template with provided kwargs."""
        return self.template.format(**kwargs)


@pytest.fixture(autouse=True)
def mock_api_key():
    """Mock the API key to avoid errors during agent creation."""
    with patch("summit_sim.agents.utils.settings.openrouter_api_key", "test-api-key"):
        yield


@pytest.fixture(autouse=True)
def clear_agent_cache():
    """Clear the agent cache before each test."""
    agent_utils._agent_container.clear()
    yield
    agent_utils._agent_container.clear()


@pytest.fixture
def mock_mlflow_prompts():
    """Mock MLflow prompt loading and registration."""

    def _create_mock(
        system_prompt: str = "Test system prompt",
        user_prompt_template: str = "Test user prompt with {{variable}}",
    ):
        system_mock = MockPrompt(system_prompt)
        user_mock = MockPrompt(user_prompt_template)

        def mock_load_prompt(uri: str):
            if "system" in uri:
                return system_mock
            if "user" in uri:
                return user_mock
            raise Exception(f"Unknown prompt URI: {uri}")

        with (
            patch("summit_sim.agents.utils.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.utils.mlflow.genai.register_prompt") as mock_reg,
        ):
            mock_load.side_effect = mock_load_prompt
            yield mock_load, mock_reg

    return _create_mock


@pytest.fixture
def mock_agent():
    """Create a mock PydanticAI agent for testing."""

    def _create_mock():
        mock_agent_instance = AsyncMock()
        with patch("summit_sim.agents.utils.Agent") as mock_agent_class:
            mock_agent_class.return_value = mock_agent_instance
            yield mock_agent_instance

    return _create_mock


@pytest.fixture
def mock_generator_prompts():
    """Mock prompts specific to the generator agent."""
    user_prompt_template = (
        "Test user prompt with {{primary_focus}} "
        "{{environment}} {{available_personnel}} "
        "{{evac_distance}} {{complexity}}"
    )
    return MockPrompt(user_prompt_template)


@pytest.fixture
def mock_simulation_prompts():
    """Mock prompts specific to the simulation agent."""
    user_prompt_template = (
        "Test user prompt with {{title}} {{setting}} {{patient_summary}} "
        "{{hidden_truth}} {{learning_objectives}} {{narrative_text}} "
        "{{choices_text}} {{selected_choice_id}} {{selected_choice_description}}"
    )
    return MockPrompt(user_prompt_template)


@pytest.fixture
def mock_debrief_prompts():
    """Mock prompts specific to the debrief agent."""
    user_prompt_template = (
        "Test user prompt with {{scenario_context}} {{scenario_id}} {{total_turns}} "
        "{{transcript_summary}} {{correct_count}} {{incorrect_count}} {{score}}"
    )
    return MockPrompt(user_prompt_template)
