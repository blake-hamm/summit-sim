"""Tests for the teacher review graph workflow."""

from typing import Any
from unittest.mock import patch

import pytest
from langgraph.types import Command

from summit_sim.agents import config as agent_config
from summit_sim.graphs.teacher_review import (
    TeacherReviewState,
    create_teacher_review_graph,
    initialize_teacher_session,
    present_for_review,
)
from summit_sim.schemas import (
    ChoiceOption,
    ScenarioDraft,
    ScenarioTurn,
    TeacherConfig,
)


@pytest.fixture
def sample_teacher_config():
    """Create a sample teacher configuration for testing."""
    return TeacherConfig(
        num_participants=3,
        activity_type="hiking",
        difficulty="med",
    )


@pytest.fixture
def sample_scenario():
    """Create a sample scenario for testing."""
    return ScenarioDraft(
        title="Test Emergency",
        setting="Mountain trail",
        patient_summary="30yo unconscious hiker",
        hidden_truth="Severe dehydration",
        learning_objectives=["Assess ABCs", "Call for help"],
        turns=[
            ScenarioTurn(
                turn_id=0,
                narrative_text="Patient is unconscious.",
                choices=[
                    ChoiceOption(
                        choice_id="check_airway",
                        description="Check airway",
                        is_correct=True,
                        next_turn_id=1,
                    ),
                    ChoiceOption(
                        choice_id="ignore",
                        description="Ignore and continue hiking",
                        is_correct=False,
                        next_turn_id=1,
                    ),
                ],
                is_starting_turn=True,
            )
        ],
        starting_turn_id=0,
    )


@pytest.fixture
def initial_state(sample_teacher_config):
    """Create initial state for testing."""
    return TeacherReviewState(
        teacher_config=sample_teacher_config.model_dump(),
        scenario_draft=None,
        scenario_id="",
        class_id="",
        retry_count=0,
        feedback_history=[],
        approval_status=None,
    )


class TestInitializeTeacherSession:
    """Tests for initialize_teacher_session node."""

    def test_initialize_generates_ids(self, initial_state):
        """Test that initialization generates scenario_id and class_id."""
        result = initialize_teacher_session(initial_state)

        assert result.scenario_id.startswith("scn-")
        assert len(result.class_id) == 6
        assert result.retry_count == 0
        assert result.feedback_history == []

    def test_initialize_preserves_teacher_config(self, initial_state):
        """Test that initialization preserves the teacher config."""
        result = initialize_teacher_session(initial_state)

        assert result.teacher_config == initial_state.teacher_config


class TestPresentForReview:
    """Tests for present_for_review node."""

    def test_present_for_review_valid(self, sample_scenario):
        """Test presenting a valid scenario for review."""
        state = TeacherReviewState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            feedback_history=[],
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.teacher_review.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.teacher_review.mlflow") as mock_mlflow,
        ):
            mock_interrupt.return_value = {"decision": "approve"}
            mock_mlflow.get_last_active_trace_id.return_value = "trace-123"
            result = present_for_review(state)

        assert result["approval_status"] == "approved"

    def test_present_for_review_no_scenario(self):
        """Test presenting with no scenario raises error."""
        state = TeacherReviewState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=None,
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            feedback_history=[],
            approval_status=None,
        )

        with pytest.raises(ValueError, match="No scenario draft available"):
            present_for_review(state)

    def test_present_for_review_invalid_decision(self, sample_scenario):
        """Test presenting with invalid decision raises error."""
        state = TeacherReviewState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            feedback_history=[],
            approval_status=None,
        )

        with patch("summit_sim.graphs.teacher_review.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"decision": "decline"}
            with pytest.raises(ValueError, match="Invalid decision: decline"):
                present_for_review(state)


class TestTeacherReviewGraphFullCycle:
    """Integration tests for full teacher review cycle."""

    @pytest.fixture(autouse=True)
    def mock_api_key(self):
        """Mock the API key to avoid errors during agent creation."""
        with patch(
            "summit_sim.agents.config.settings.openrouter_api_key", "test-api-key"
        ):
            yield

    @pytest.fixture(autouse=True)
    def mock_mlflow_span(self):
        """Mock MLflow span to avoid trace_id errors."""
        mock_span = type("MockSpan", (), {"trace_id": "test-trace-123"})()
        with (
            patch(
                "summit_sim.graphs.teacher_review.mlflow.get_current_active_span",
                return_value=mock_span,
            ),
            patch(
                "summit_sim.graphs.teacher_review.mlflow.log_feedback",
            ),
        ):
            yield

    @pytest.fixture(autouse=True)
    def clear_agent_cache(self):
        """Clear the agent cache before each test."""
        agent_config._agent_container.clear()

    @pytest.fixture(autouse=True)
    def mock_prompts(self):
        """Mock MLflow prompt loading."""

        class MockPrompt:
            def __init__(self, template):
                self.template = template

            def format(self, **kwargs):
                return self.template.format(**kwargs)

        user_prompt = (
            "Test user prompt with {num_participants} {activity_type} {difficulty}"
        )
        mock_prompt_obj = MockPrompt(user_prompt)

        with (
            patch("summit_sim.agents.config.mlflow.genai.load_prompt") as mock_load,
            patch("summit_sim.agents.config.mlflow.genai.register_prompt"),
            patch(
                "summit_sim.agents.generator.mlflow.genai.load_prompt"
            ) as mock_load_gen,
        ):
            mock_load.return_value = MockPrompt("Test system prompt")
            mock_load_gen.return_value = mock_prompt_obj
            yield

    @pytest.mark.asyncio
    async def test_full_happy_path(self, sample_teacher_config, sample_scenario):
        """Test complete teacher review happy path with mocked generator."""

        async def mock_generate_impl(_config):
            """Mock implementation that returns the sample scenario."""
            return sample_scenario

        with patch(
            "summit_sim.graphs.teacher_review.generate_scenario"
        ) as mock_generate:
            mock_generate.side_effect = mock_generate_impl

            with patch("summit_sim.graphs.teacher_review.interrupt") as mock_interrupt:
                mock_interrupt.return_value = {"decision": "approve"}

                initial_state = TeacherReviewState(
                    teacher_config=sample_teacher_config.model_dump(),
                    scenario_draft=None,
                    scenario_id="",
                    class_id="",
                    retry_count=0,
                    feedback_history=[],
                    approval_status=None,
                )

                graph = create_teacher_review_graph()
                config: Any = {"configurable": {"thread_id": "test-thread"}}

                # Run to interrupt point
                result = await graph.ainvoke(initial_state, config)

                # Verify we reached the interrupt
                assert result.get("scenario_draft") is not None
                assert result.get("scenario_id", "").startswith("scn-")
                assert len(result.get("class_id", "")) == 6

                # Resume with approval
                final_result = await graph.ainvoke(
                    Command(resume={"decision": "approve"}), config
                )

                # Verify approval
                assert final_result.get("approval_status") == "approved"

    def test_graph_creation(self):
        """Test that graph can be created successfully."""
        graph = create_teacher_review_graph()
        assert graph is not None
