"""Tests for the teacher review graph workflow."""

from typing import Any
from unittest.mock import patch

import pytest
from langgraph.types import Command

from summit_sim.agents import config as agent_config
from summit_sim.graphs.teacher import (
    TeacherState,
    create_teacher_graph,
    initialize_teacher,
    present_for_teacher,
    should_retry,
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
    return TeacherState(
        teacher_config=sample_teacher_config.model_dump(),
        scenario_draft=None,
        scenario_id="",
        class_id="",
        retry_count=0,
        approval_status=None,
    )


class TestInitializeTeacherSession:
    """Tests for initialize_teacher node."""

    def test_initialize_generates_ids(self, initial_state):
        """Test that initialization generates scenario_id and class_id."""
        result = initialize_teacher(initial_state)

        assert result.scenario_id.startswith("scn-")
        assert len(result.class_id) == 6
        assert result.retry_count == 0

    def test_initialize_preserves_teacher_config(self, initial_state):
        """Test that initialization preserves the teacher config."""
        result = initialize_teacher(initial_state)

        assert result.teacher_config == initial_state.teacher_config


class TestPresentForReview:
    """Tests for present_for_teacher node."""

    def test_present_for_teacher_valid_rating_5(self, sample_scenario):
        """Test presenting a valid scenario with rating 5."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.teacher.mlflow.log_feedback") as mock_log,
        ):
            mock_interrupt.return_value = {"rating": 5}
            result = present_for_teacher(state)

        assert result["approval_status"] == "approved"
        assert result["teacher_rating"] == 5
        mock_log.assert_called_once()

    def test_present_for_teacher_rating_3_acceptable(self, sample_scenario):
        """Test rating 3 is acceptable and approved."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.teacher.mlflow.log_feedback") as mock_log,
        ):
            mock_interrupt.return_value = {"rating": 3}
            result = present_for_teacher(state)

        assert result["approval_status"] == "approved"
        assert result["teacher_rating"] == 3
        mock_log.assert_called_once()

    def test_present_for_teacher_rating_2_rejected(self, sample_scenario):
        """Test rating 2 is rejected and should trigger retry."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.teacher.mlflow.log_feedback") as mock_log,
        ):
            mock_interrupt.return_value = {"rating": 2}
            result = present_for_teacher(state)

        assert result["approval_status"] == "rejected"
        assert result["teacher_rating"] == 2
        mock_log.assert_called_once()

    def test_present_for_teacher_no_scenario(self):
        """Test presenting with no scenario raises error."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=None,
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
        )

        with pytest.raises(ValueError, match="No scenario draft available"):
            present_for_teacher(state)

    def test_present_for_teacher_invalid_rating(self, sample_scenario):
        """Test presenting with invalid rating raises error."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
        )

        with patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"rating": 6}
            with pytest.raises(ValueError, match="Invalid rating: 6"):
                present_for_teacher(state)

    def test_present_for_teacher_missing_rating(self, sample_scenario):
        """Test presenting with missing rating raises error."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
        )

        with patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"decision": "approve"}
            with pytest.raises(ValueError, match="Invalid rating: None"):
                present_for_teacher(state)


class TestShouldRetryLogic:
    """Tests for should_retry conditional edge."""

    def test_should_retry_low_rating_under_limit(self, sample_scenario):
        """Test should_retry routes to generate when rating < 3 and retry_count < 3."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=1,
            approval_status=None,
            teacher_rating=2,
        )

        result = should_retry(state)
        assert result == "generate"

    def test_should_retry_low_rating_at_limit(self, sample_scenario):
        """Test should_retry routes to save when rating < 3 but retry_count >= 3."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=3,
            approval_status=None,
            teacher_rating=2,
        )

        result = should_retry(state)
        assert result == "save"

    def test_should_retry_acceptable_rating(self, sample_scenario):
        """Test should_retry routes to save when rating >= 3."""
        state = TeacherState(
            teacher_config=TeacherConfig(
                num_participants=3, activity_type="hiking", difficulty="med"
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            class_id="abc123",
            retry_count=0,
            approval_status=None,
            teacher_rating=4,
        )

        result = should_retry(state)
        assert result == "save"


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
                "summit_sim.graphs.teacher.mlflow.get_current_active_span",
                return_value=mock_span,
            ),
            patch(
                "summit_sim.graphs.teacher.mlflow.log_feedback",
            ),
            patch(
                "summit_sim.agents.generator.mlflow.get_current_active_span",
                return_value=mock_span,
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
    async def test_full_happy_path_rating_5(
        self, sample_teacher_config, sample_scenario
    ):
        """Test complete teacher review happy path with rating 5."""

        async def mock_generate_impl(_config):
            """Mock implementation that returns the sample scenario."""
            return sample_scenario

        with patch("summit_sim.graphs.teacher.generate_scenario") as mock_generate:
            mock_generate.side_effect = mock_generate_impl

            with patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt:
                mock_interrupt.return_value = {
                    "rating": 5,
                }

                initial_state = TeacherState(
                    teacher_config=sample_teacher_config.model_dump(),
                    scenario_draft=None,
                    scenario_id="",
                    class_id="",
                    retry_count=0,
                    approval_status=None,
                )

                graph = create_teacher_graph()
                config: Any = {"configurable": {"thread_id": "test-thread"}}

                # Run to interrupt point
                result = await graph.ainvoke(initial_state, config)

                # Verify we reached the interrupt
                assert result.get("scenario_draft") is not None
                assert result.get("scenario_id", "").startswith("scn-")
                assert len(result.get("class_id", "")) == 6

                # Resume with rating 5
                final_result = await graph.ainvoke(
                    Command(
                        resume={
                            "rating": 5,
                        }
                    ),
                    config,
                )

                # Verify approval
                assert final_result.get("approval_status") == "approved"
                assert final_result.get("teacher_rating") == 5

    @pytest.mark.asyncio
    async def test_retry_flow_rating_2_then_5(
        self, sample_teacher_config, sample_scenario
    ):
        """Test retry flow: rating 2 triggers regeneration, then rating 5 succeeds."""
        call_count = 0

        async def mock_generate_impl(_config):
            """Mock implementation that returns the sample scenario."""
            nonlocal call_count
            call_count += 1
            return sample_scenario

        with patch("summit_sim.graphs.teacher.generate_scenario") as mock_generate:
            mock_generate.side_effect = mock_generate_impl

            with patch("summit_sim.graphs.teacher.interrupt") as mock_interrupt:
                # First call returns rating 2, second call returns rating 5
                mock_interrupt.side_effect = [
                    {
                        "rating": 2,
                    },
                    {
                        "rating": 5,
                    },
                ]

                initial_state = TeacherState(
                    teacher_config=sample_teacher_config.model_dump(),
                    scenario_draft=None,
                    scenario_id="",
                    class_id="",
                    retry_count=0,
                    approval_status=None,
                )

                graph = create_teacher_graph()
                config: Any = {"configurable": {"thread_id": "test-thread"}}

                # First generation
                result = await graph.ainvoke(initial_state, config)
                assert result.get("scenario_draft") is not None

                # Resume with rating 2 - should trigger retry
                result_after_rating = await graph.ainvoke(
                    Command(
                        resume={
                            "rating": 2,
                        }
                    ),
                    config,
                )

                # Should have regenerated
                assert call_count == 2
                assert result_after_rating.get("retry_count") == 1

                # Resume with rating 5 - should complete
                final_result = await graph.ainvoke(
                    Command(
                        resume={
                            "rating": 5,
                        }
                    ),
                    config,
                )

                assert final_result.get("approval_status") == "approved"
                assert final_result.get("teacher_rating") == 5

    def test_graph_creation(self):
        """Test that graph can be created successfully."""
        graph = create_teacher_graph()
        assert graph is not None
