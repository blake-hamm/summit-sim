"""Tests for the author review graph workflow."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.types import Command

from summit_sim.agents import utils as agent_utils
from summit_sim.graphs.author import (
    AuthorState,
    create_author_graph,
    initialize_author,
    present_for_author,
    should_retry,
)
from summit_sim.schemas import (
    ChoiceOption,
    ScenarioConfig,
    ScenarioDraft,
    ScenarioTurn,
)
from summit_sim.ui import author as author_ui


def _session_get_participants(key: str) -> str | None:
    return {
        "primary_focus": "Trauma",
        "environment": "Alpine/Mountain",
        "available_personnel": "Small Group (3-5)",
        "evac_distance": "Remote (1 day)",
        "complexity": "Standard",
    }.get(key)


def _session_get_thread(key: str) -> Any:
    return {"id": "test-thread", "graph": None}.get(key)


def _session_get(key: str, data: dict) -> Any:
    return data.get(key)


def _session_get_for_rating(key: str, mock_graph: AsyncMock | None = None) -> Any:
    return {"id": "test-thread", "graph": mock_graph}.get(key)


@pytest.fixture
def sample_scenario_config():
    """Create a sample teacher configuration for testing."""
    return ScenarioConfig(
        primary_focus="Trauma",
        environment="Alpine/Mountain",
        available_personnel="Small Group (3-5)",
        evac_distance="Remote (1 day)",
        complexity="Standard",
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
                    ChoiceOption(
                        choice_id="panic",
                        description="Panic",
                        is_correct=False,
                        next_turn_id=1,
                    ),
                ],
            ),
            ScenarioTurn(
                turn_id=1,
                narrative_text="Airway is clear but breathing is shallow.",
                choices=[
                    ChoiceOption(
                        choice_id="check_breathing",
                        description="Check breathing rate",
                        is_correct=True,
                        next_turn_id=2,
                    ),
                    ChoiceOption(
                        choice_id="move_patient",
                        description="Move patient to shade",
                        is_correct=False,
                        next_turn_id=2,
                    ),
                    ChoiceOption(
                        choice_id="give_water",
                        description="Give water",
                        is_correct=False,
                        next_turn_id=2,
                    ),
                ],
            ),
            ScenarioTurn(
                turn_id=2,
                narrative_text="Patient has weak pulse.",
                choices=[
                    ChoiceOption(
                        choice_id="call_911",
                        description="Call emergency services",
                        is_correct=True,
                        next_turn_id=None,
                    ),
                    ChoiceOption(
                        choice_id="wait",
                        description="Wait and observe",
                        is_correct=False,
                        next_turn_id=None,
                    ),
                    ChoiceOption(
                        choice_id="cpr",
                        description="Begin CPR",
                        is_correct=False,
                        next_turn_id=None,
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def initial_state(sample_scenario_config):
    """Create initial state for testing."""
    return AuthorState(
        scenario_config=sample_scenario_config.model_dump(),
        scenario_draft=None,
        scenario_id="",
        retry_count=0,
        approval_status=None,
    )


class TestInitializeTeacherSession:
    """Tests for initialize_author node."""

    def test_initialize_generates_ids(self, initial_state):
        """Test that initialization generates scenario_id."""
        result = initialize_author(initial_state)

        assert result.scenario_id.startswith("scn-")
        assert result.retry_count == 0

    def test_initialize_preserves_scenario_config(self, initial_state):
        """Test that initialization preserves the teacher config."""
        result = initialize_author(initial_state)

        assert result.scenario_config == initial_state.scenario_config


class TestPresentForReview:
    """Tests for present_for_author node."""

    def test_present_for_author_valid_rating_5(self, sample_scenario):
        """Test presenting a valid scenario with rating 5."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.author.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.author.mlflow.log_feedback") as mock_log,
        ):
            mock_interrupt.return_value = {"rating": 5}
            result = present_for_author(state)

        assert result["approval_status"] == "approved"
        assert result["author_rating"] == 5
        mock_log.assert_called_once()

    def test_present_for_author_rating_3_acceptable(self, sample_scenario):
        """Test rating 3 is acceptable and approved."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.author.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.author.mlflow.log_feedback") as mock_log,
        ):
            mock_interrupt.return_value = {"rating": 3}
            result = present_for_author(state)

        assert result["approval_status"] == "approved"
        assert result["author_rating"] == 3
        mock_log.assert_called_once()

    def test_present_for_author_rating_2_rejected(self, sample_scenario):
        """Test rating 2 is rejected and should trigger retry."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
            current_trace_id="trace-123",
        )

        with (
            patch("summit_sim.graphs.author.interrupt") as mock_interrupt,
            patch("summit_sim.graphs.author.mlflow.log_feedback") as mock_log,
        ):
            mock_interrupt.return_value = {"rating": 2}
            result = present_for_author(state)

        assert result["approval_status"] == "rejected"
        assert result["author_rating"] == 2
        mock_log.assert_called_once()

    def test_present_for_author_no_scenario(self):
        """Test presenting with no scenario raises error."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=None,
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )

        with pytest.raises(ValueError, match="No scenario draft available"):
            present_for_author(state)

    def test_present_for_author_invalid_rating(self, sample_scenario):
        """Test presenting with invalid rating raises error."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"rating": 6}
            with pytest.raises(ValueError, match="Invalid rating: 6"):
                present_for_author(state)

    def test_present_for_author_missing_rating(self, sample_scenario):
        """Test presenting with missing rating raises error."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"decision": "approve"}
            with pytest.raises(ValueError, match="Invalid rating: None"):
                present_for_author(state)


class TestShouldRetryLogic:
    """Tests for should_retry conditional edge."""

    def test_should_retry_low_rating_under_limit(self, sample_scenario):
        """Test should_retry routes to generate when rating < 3 and retry_count < 3."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=1,
            approval_status=None,
            author_rating=2,
        )

        result = should_retry(state)
        assert result == "generate"

    def test_should_retry_low_rating_at_limit(self, sample_scenario):
        """Test should_retry routes to save when rating < 3 but retry_count >= 3."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=3,
            approval_status=None,
            author_rating=2,
        )

        result = should_retry(state)
        assert result == "save"

    def test_should_retry_acceptable_rating(self, sample_scenario):
        """Test should_retry routes to save when rating >= 3."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
            author_rating=4,
        )

        result = should_retry(state)
        assert result == "save"


class TestTeacherReviewGraphFullCycle:
    """Integration tests for full teacher review cycle."""

    @pytest.fixture(autouse=True)
    def mock_api_key(self):
        """Mock the API key to avoid errors during agent creation."""
        with patch(
            "summit_sim.agents.utils.settings.openrouter_api_key", "test-api-key"
        ):
            yield

    @pytest.fixture(autouse=True)
    def mock_mlflow_span(self):
        """Mock MLflow span to avoid trace_id errors."""
        mock_span = type("MockSpan", (), {"trace_id": "test-trace-123"})()
        with (
            patch(
                "summit_sim.graphs.author.mlflow.get_current_active_span",
                return_value=mock_span,
            ),
            patch(
                "summit_sim.graphs.author.mlflow.log_feedback",
            ),
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
    async def test_full_happy_path_rating_5(
        self, sample_scenario_config, sample_scenario
    ):
        """Test complete teacher review happy path with rating 5."""

        async def mock_generate_impl(_config):
            """Mock implementation that returns the sample scenario."""
            return sample_scenario

        with patch("summit_sim.graphs.author.generate_scenario") as mock_generate:
            mock_generate.side_effect = mock_generate_impl

            with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
                mock_interrupt.return_value = {
                    "rating": 5,
                }

                initial_state = AuthorState(
                    scenario_config=sample_scenario_config.model_dump(),
                    scenario_draft=None,
                    scenario_id="",
                    retry_count=0,
                    approval_status=None,
                )

                graph = create_author_graph()
                config: Any = {"configurable": {"thread_id": "test-thread"}}

                # Run to interrupt point
                result = await graph.ainvoke(initial_state, config)

                # Verify we reached the interrupt
                assert result.get("scenario_draft") is not None
                assert result.get("scenario_id", "").startswith("scn-")
                assert result.get("scenario_id", "").startswith("scn-")

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
                assert final_result.get("author_rating") == 5

    @pytest.mark.asyncio
    async def test_retry_flow_rating_2_then_5(
        self, sample_scenario_config, sample_scenario
    ):
        """Test retry flow: rating 2 triggers regeneration, then rating 5 succeeds."""
        call_count = 0

        async def mock_generate_impl(_config):
            """Mock implementation that returns the sample scenario."""
            nonlocal call_count
            call_count += 1
            return sample_scenario

        with patch("summit_sim.graphs.author.generate_scenario") as mock_generate:
            mock_generate.side_effect = mock_generate_impl

            with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
                # First call returns rating 2, second call returns rating 5
                mock_interrupt.side_effect = [
                    {
                        "rating": 2,
                    },
                    {
                        "rating": 5,
                    },
                ]

                initial_state = AuthorState(
                    scenario_config=sample_scenario_config.model_dump(),
                    scenario_draft=None,
                    scenario_id="",
                    retry_count=0,
                    approval_status=None,
                )

                graph = create_author_graph()
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
                assert final_result.get("author_rating") == 5

    def test_graph_creation(self):
        """Test that graph can be created successfully."""
        graph = create_author_graph()
        assert graph is not None


class TestGenerateScenario:
    """Tests for the generate_scenario UI function."""

    @pytest.mark.asyncio
    async def test_generate_scenario_success(self, sample_scenario_config):
        """Test successful scenario generation flow."""
        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=_session_get_participants,
            ),
            patch.object(author_ui.cl.user_session, "set"),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(
                author_ui, "create_author_graph", return_value=mock_graph
            ) as mock_create_graph,
            patch.object(author_ui, "show_review_screen", new_callable=AsyncMock),
        ):
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "scenario_config": sample_scenario_config.model_dump(),
                    "scenario_draft": {
                        "title": "Test",
                        "setting": "Mountain",
                        "patient_summary": "Test patient",
                        "hidden_truth": "truth",
                        "learning_objectives": [],
                        "turns": [],
                        "starting_turn_id": 0,
                    },
                    "scenario_id": "scn-test",
                    "class_id": "cls-123",
                    "retry_count": 0,
                }
            )

            await author_ui.generate_scenario()

            mock_create_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_scenario_missing_config_raises(self):
        """Test scenario generation raises when session values are missing."""
        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(author_ui.cl.user_session, "get", return_value=None),
            patch.object(author_ui.cl.user_session, "set"),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(author_ui, "create_author_graph", return_value=mock_graph),
        ):
            with pytest.raises(ValueError, match="Missing required scenario config"):
                await author_ui.generate_scenario()

    @pytest.mark.asyncio
    async def test_generate_scenario_failure_no_draft(self):
        """Test scenario generation handles missing draft."""
        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=_session_get_participants,
            ),
            patch.object(author_ui.cl.user_session, "set"),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(author_ui, "create_author_graph", return_value=mock_graph),
        ):
            mock_graph.ainvoke = AsyncMock(return_value={})

            await author_ui.generate_scenario()

    @pytest.mark.asyncio
    async def test_generate_scenario_exception(self):
        """Test scenario generation handles exceptions."""
        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=_session_get_participants,
            ),
            patch.object(author_ui.cl.user_session, "set"),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(author_ui, "create_author_graph", return_value=mock_graph),
        ):
            mock_graph.ainvoke = AsyncMock(side_effect=Exception("API Error"))

            await author_ui.generate_scenario()


class TestShowReviewScreen:
    """Tests for the show_review_screen UI function."""

    @pytest.mark.asyncio
    async def test_show_review_screen_success(self, sample_scenario):
        """Test successful review screen display."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )

        mock_response = {"payload": {"value": 4}}
        mock_message = AsyncMock()

        with (
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(
                author_ui.cl,
                "AskActionMessage",
                return_value=AsyncMock(send=AsyncMock(return_value=mock_response)),
            ) as mock_ask,
            patch.object(author_ui, "handle_rating", new_callable=AsyncMock),
        ):
            await author_ui.show_review_screen(state)

            mock_message.send.assert_called()
            mock_ask.assert_called()

    @pytest.mark.asyncio
    async def test_show_review_screen_with_retry(self, sample_scenario):
        """Test review screen shows retry attempt text."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=1,
            approval_status=None,
        )
        mock_message = AsyncMock()

        with (
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(
                author_ui.cl,
                "AskActionMessage",
                return_value=AsyncMock(send=AsyncMock(return_value=None)),
            ),
        ):
            await author_ui.show_review_screen(state)

            mock_message.send.assert_called()

    @pytest.mark.asyncio
    async def test_show_review_screen_no_scenario(self):
        """Test review screen handles no scenario."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=None,
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )
        mock_message = AsyncMock()

        with patch.object(author_ui.cl, "Message", return_value=mock_message):
            await author_ui.show_review_screen(state)

            mock_message.send.assert_called()

    @pytest.mark.asyncio
    async def test_show_review_screen_no_rating(self, sample_scenario):
        """Test review screen handles no rating response."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )
        mock_message = AsyncMock()

        with (
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(
                author_ui.cl,
                "AskActionMessage",
                return_value=AsyncMock(send=AsyncMock(return_value={})),
            ),
            patch.object(author_ui, "handle_rating", new_callable=AsyncMock),
        ):
            await author_ui.show_review_screen(state)


class TestHandleRating:
    """Tests for the handle_rating UI function."""

    @pytest.mark.asyncio
    async def test_handle_rating_approved(self, sample_scenario):
        """Test rating >= 3 leads to completion."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
            author_rating=4,
        )

        mock_graph = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=lambda k: _session_get_for_rating(k, mock_graph),
            ),
            patch.object(author_ui, "show_completion", new_callable=AsyncMock),
        ):
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "scenario_config": state.scenario_config,
                    "scenario_draft": sample_scenario.model_dump(),
                    "scenario_id": "scn-test123",
                    "class_id": "abc123",
                    "retry_count": 0,
                    "approval_status": "approved",
                    "author_rating": 4,
                }
            )

            await author_ui.handle_rating(state, 4)

    @pytest.mark.asyncio
    async def test_handle_rating_retry(self, sample_scenario):
        """Test rating < 3 triggers regeneration."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
            author_rating=2,
        )

        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=lambda k: _session_get_for_rating(k, mock_graph),
            ),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(author_ui, "show_review_screen", new_callable=AsyncMock),
        ):
            mock_graph.ainvoke = AsyncMock(
                side_effect=[
                    {
                        "scenario_config": state.scenario_config,
                        "scenario_draft": sample_scenario.model_dump(),
                        "scenario_id": "scn-test123",
                        "class_id": "abc123",
                        "retry_count": 1,
                        "approval_status": "rejected",
                        "author_rating": 2,
                    },
                    {
                        "scenario_config": state.scenario_config,
                        "scenario_draft": sample_scenario.model_dump(),
                        "scenario_id": "scn-test123",
                        "class_id": "abc123",
                        "retry_count": 1,
                    },
                ]
            )

            await author_ui.handle_rating(state, 2)

            mock_message.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_rating_max_retries(self, sample_scenario):
        """Test rating < 3 at max retries proceeds anyway."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=2,
            approval_status=None,
            author_rating=2,
        )

        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=lambda k: _session_get_for_rating(k, mock_graph),
            ),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(author_ui, "show_completion", new_callable=AsyncMock),
        ):
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "scenario_config": state.scenario_config,
                    "scenario_draft": sample_scenario.model_dump(),
                    "scenario_id": "scn-test123",
                    "class_id": "abc123",
                    "retry_count": 3,
                    "approval_status": "rejected",
                    "author_rating": 2,
                }
            )

            await author_ui.handle_rating(state, 2)

            mock_message.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_rating_no_graph(self, sample_scenario):
        """Test handling when graph is missing from session."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )

        mock_message = AsyncMock()

        with (
            patch.object(author_ui.cl.user_session, "get", return_value=None),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
        ):
            await author_ui.handle_rating(state, 4)

            mock_message.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_rating_exception(self, sample_scenario):
        """Test handling exceptions during rating."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=sample_scenario.model_dump(),
            scenario_id="scn-test123",
            retry_count=0,
            approval_status=None,
        )

        mock_graph = AsyncMock()
        mock_message = AsyncMock()

        with (
            patch.object(
                author_ui.cl.user_session,
                "get",
                side_effect=lambda k: _session_get_for_rating(k, mock_graph),
            ),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
            patch.object(author_ui, "show_review_screen", new_callable=AsyncMock),
        ):
            mock_graph.ainvoke = AsyncMock(side_effect=Exception("API Error"))

            await author_ui.handle_rating(state, 4)

            mock_message.send.assert_called()


class TestShowCompletion:
    """Tests for the show_completion UI function."""

    @pytest.mark.asyncio
    async def test_show_completion(self):
        """Test completion screen displays shareable link."""
        state = AuthorState(
            scenario_config=ScenarioConfig(
                primary_focus="Trauma",
                environment="Alpine/Mountain",
                available_personnel="Small Group (3-5)",
                evac_distance="Remote (1 day)",
                complexity="Standard",
            ).model_dump(),
            scenario_draft=None,
            scenario_id="scn-test123",
            retry_count=0,
            approval_status="approved",
        )
        mock_message = AsyncMock()

        with (
            patch.object(author_ui.settings, "base_url", "https://example.com"),
            patch.object(author_ui.cl, "Message", return_value=mock_message),
        ):
            await author_ui.show_completion(state)

            mock_message.send.assert_called()


class TestAskScenarioConfig:
    """Tests for the ask_scenario_config UI function."""

    @pytest.mark.asyncio
    async def test_ask_scenario_config_success(self):
        """Test successful scenario configuration with form submission."""
        mock_response = {
            "submitted": True,
            "primary_focus": "Trauma",
            "environment": "Alpine/Mountain",
            "available_personnel": "Small Group (3-5)",
            "evac_distance": "Remote (1 day)",
            "complexity": "Standard",
        }

        mock_element = type("MockElement", (), {"name": "ScenarioConfigForm"})()

        with (
            patch.object(
                author_ui.cl,
                "CustomElement",
                return_value=mock_element,
            ),
            patch.object(
                author_ui.cl,
                "AskElementMessage",
                return_value=AsyncMock(send=AsyncMock(return_value=mock_response)),
            ) as mock_ask,
            patch.object(author_ui.cl.user_session, "set") as mock_set,
            patch.object(
                author_ui, "generate_scenario", new_callable=AsyncMock
            ) as mock_generate,
        ):
            await author_ui.ask_scenario_config()

            # Verify the form was displayed
            mock_ask.assert_called_once()
            call_args = mock_ask.call_args
            assert "Configure Your Scenario" in call_args.kwargs["content"]

            # Verify session values were set
            mock_set.assert_any_call("primary_focus", "Trauma")
            mock_set.assert_any_call("environment", "Alpine/Mountain")
            mock_set.assert_any_call("available_personnel", "Small Group (3-5)")

            # Verify scenario generation was called
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_scenario_config_cancelled(self):
        """Test that cancelled form does not proceed."""
        mock_response = {
            "submitted": False,
        }

        mock_element = type("MockElement", (), {"name": "ScenarioConfigForm"})()

        with (
            patch.object(
                author_ui.cl,
                "CustomElement",
                return_value=mock_element,
            ),
            patch.object(
                author_ui.cl,
                "AskElementMessage",
                return_value=AsyncMock(send=AsyncMock(return_value=mock_response)),
            ),
            patch.object(author_ui.cl.user_session, "set") as mock_set,
            patch.object(
                author_ui, "generate_scenario", new_callable=AsyncMock
            ) as mock_generate,
        ):
            await author_ui.ask_scenario_config()

            # Verify no session values were set
            mock_set.assert_not_called()
            # Verify scenario generation was not called
            mock_generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_scenario_config_no_response(self):
        """Test that no response (None) does not proceed."""
        mock_element = type("MockElement", (), {"name": "ScenarioConfigForm"})()

        with (
            patch.object(
                author_ui.cl,
                "CustomElement",
                return_value=mock_element,
            ),
            patch.object(
                author_ui.cl,
                "AskElementMessage",
                return_value=AsyncMock(send=AsyncMock(return_value=None)),
            ),
            patch.object(author_ui.cl.user_session, "set") as mock_set,
            patch.object(
                author_ui, "generate_scenario", new_callable=AsyncMock
            ) as mock_generate,
        ):
            await author_ui.ask_scenario_config()

            # Verify no session values were set
            mock_set.assert_not_called()
            # Verify scenario generation was not called
            mock_generate.assert_not_called()
