"""Pytest configuration and shared fixtures for summit-sim tests."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summit_sim.agents import utils as agent_utils

# Disable MLflow tracing entirely in tests to prevent database creation
os.environ["MLFLOW_TRACING_ENABLE"] = "false"
# Use in-memory SQLite to avoid creating mlflow.db file
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///:memory:"
os.environ["MLFLOW_ARTIFACT_URI"] = "file:///tmp/mlflow_test_artifacts"
# Disable PydanticAI autologging which creates its own database
os.environ["MLFLOW_PYDANTICAI_AUTOLOG_DISABLED"] = "true"


class MockPrompt:
    """Mock MLflow prompt object for testing."""

    def __init__(self, template: str) -> None:
        """Initialize mock prompt with template."""
        self.template = template

    def format(self, **kwargs) -> str:
        """Format the template with provided kwargs."""
        return self.template.format(**kwargs)


@pytest.fixture(autouse=True)
def mock_gcp_credentials():
    """Mock the GCP credentials to avoid errors during agent creation."""
    with (
        patch(
            "summit_sim.agents.utils.settings.google_application_credentials",
            Path("/tmp/test-sa.json"),
        ),
        patch("summit_sim.agents.utils.settings.gcp_project_id", "test-project"),
        patch("summit_sim.agents.utils.settings.gcp_location", "us-central1"),
        patch("summit_sim.agents.utils.service_account.Credentials") as mock_creds,
    ):
        mock_creds.from_service_account_file.return_value = MagicMock()
        yield


@pytest.fixture(autouse=True)
def _clear_agent_cache():
    """Clear the agent cache before each test."""
    agent_utils._agent_container.clear()
    yield
    agent_utils._agent_container.clear()


@pytest.fixture(autouse=True)
def mock_mlflow():
    """Mock MLflow tracing and prompt registry globally for all tests."""
    with (
        patch("summit_sim.graphs.author.mlflow") as mock_mlflow_author,
        patch("summit_sim.graphs.simulation.mlflow") as mock_mlflow_simulation,
        patch("summit_sim.agents.utils.mlflow") as mock_mlflow_utils,
    ):
        # Mock trace decorator to just return the function
        def mock_trace_decorator(*args, **_kwargs):
            def decorator(func):
                return func

            if args and callable(args[0]):
                return args[0]
            return decorator

        mock_mlflow_author.trace = mock_trace_decorator
        mock_mlflow_simulation.trace = mock_trace_decorator

        # Mock SpanType enum
        class MockSpanType:
            AGENT = "AGENT"
            LLM = "LLM"
            CHAIN = "CHAIN"
            TOOL = "TOOL"

        mock_mlflow_author.SpanType = MockSpanType
        mock_mlflow_simulation.SpanType = MockSpanType

        # Mock AssessmentSourceType
        class MockAssessmentSourceType:
            LLM_JUDGE = "LLM_JUDGE"
            HUMAN = "HUMAN"
            CODE = "CODE"

        mock_mlflow_author.AssessmentSourceType = MockAssessmentSourceType

        # Mock AssessmentSource
        class MockAssessmentSource:
            def __init__(self, source_type, source_id=None):
                self.source_type = source_type
                self.source_id = source_id

        mock_mlflow_author.AssessmentSource = MockAssessmentSource

        # Mock get_current_active_span to return a mock span
        mock_span = MagicMock()
        mock_span.trace_id = "test-trace-id-123"
        mock_mlflow_author.get_current_active_span.return_value = mock_span
        mock_mlflow_simulation.get_current_active_span.return_value = mock_span

        # Mock update_current_trace
        mock_mlflow_author.update_current_trace = MagicMock()
        mock_mlflow_simulation.update_current_trace = MagicMock()

        # Mock genai.load_prompt and genai.register_prompt
        def mock_load_prompt(uri: str):
            if "system" in uri:
                return MockPrompt("Test system prompt")
            if "user" in uri:
                return MockPrompt("Test user prompt with {{variable}}")
            raise Exception(f"Unknown prompt URI: {uri}")

        mock_mlflow_utils.genai.load_prompt.side_effect = mock_load_prompt
        mock_mlflow_utils.genai.register_prompt = MagicMock()

        # Also disable MLflow trace export to prevent warnings
        with (
            patch(
                "mlflow.tracing.export.mlflow_v3.MlflowV3SpanExporter.export"
            ) as mock_export,
            patch(
                "mlflow.tracing.export.inference_table.InferenceTableSpanExporter.export"
            ) as mock_export2,
        ):
            mock_export.return_value = None
            mock_export2.return_value = None
            yield {
                "author": mock_mlflow_author,
                "simulation": mock_mlflow_simulation,
                "utils": mock_mlflow_utils,
            }


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
