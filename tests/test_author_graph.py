"""Tests for author graph workflow."""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver

from summit_sim.graphs.author import (
    ACCEPTABLE_RATING_THRESHOLD,
    MAX_RATING,
    MAX_RETRY_ATTEMPTS,
    MIN_RATING,
    AuthorState,
    create_author_graph,
    initialize_author,
    present_for_author,
    save_scenario,
    should_retry,
)
from summit_sim.schemas import ScenarioConfig, ScenarioDraft


class TestAuthorState:
    """Tests for AuthorState dataclass."""

    def test_author_state_creation(self):
        """Test creating an AuthorState instance."""
        config = ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Partner (2)",
            evac_distance="Short (< 2 hours)",
            complexity="Standard",
        )

        state = AuthorState(
            scenario_config=config.model_dump(),
            scenario_id="test-scenario-123",
            retry_count=0,
        )

        assert state.scenario_id == "test-scenario-123"
        assert state.retry_count == 0
        assert state.scenario_draft is None
        assert state.approval_status is None

    def test_author_state_defaults(self):
        """Test AuthorState default values."""
        config = ScenarioConfig(
            primary_focus="Medical",
            environment="Forest/Trail",
            available_personnel="Solo Rescuer (1)",
            evac_distance="Remote (1 day)",
            complexity="Complicated",
        )

        state = AuthorState(scenario_config=config.model_dump())

        assert state.scenario_id == ""
        assert state.retry_count == 0
        assert state.scenario_draft is None
        assert state.approval_status is None
        assert state.current_trace_id is None
        assert state.author_rating is None

    def test_from_graph_result(self):
        """Test creating state from graph result."""
        config = ScenarioConfig(
            primary_focus="Environmental",
            environment="Desert",
            available_personnel="Small Group (3-5)",
            evac_distance="Expedition (2+ days)",
            complexity="Critical",
        )

        result = {
            "scenario_config": config.model_dump(),
            "scenario_draft": {"title": "Test"},
            "scenario_id": "scn-456",
            "retry_count": 2,
            "approval_status": "approved",
            "current_trace_id": "trace-789",
            "author_rating": 5,
            "extra_field": "should be filtered",
        }

        state = AuthorState.from_graph_result(result)

        assert state.scenario_id == "scn-456"
        assert state.retry_count == 2
        assert state.approval_status == "approved"
        assert state.author_rating == 5
        assert not hasattr(state, "extra_field")


class TestInitializeAuthor:
    """Tests for initialize_author function."""

    def test_initialize_author(self):
        """Test author workflow initialization."""
        config = ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Partner (2)",
            evac_distance="Short (< 2 hours)",
            complexity="Standard",
        )

        initial_state = AuthorState(
            scenario_config=config.model_dump(),
        )

        result = initialize_author(initial_state)

        assert result.scenario_id != ""
        assert result.scenario_id.startswith("scn-")
        assert result.retry_count == 0
        assert result.scenario_config == config.model_dump()

    def test_initialize_preserves_config(self):
        """Test that config is preserved during initialization."""
        config = ScenarioConfig(
            primary_focus="Medical",
            environment="Winter/Snow",
            available_personnel="Large Expedition (6+)",
            evac_distance="Expedition (2+ days)",
            complexity="Critical",
        )

        initial_state = AuthorState(scenario_config=config.model_dump())

        result = initialize_author(initial_state)

        assert result.scenario_config["primary_focus"] == "Medical"
        assert result.scenario_config["environment"] == "Winter/Snow"


class TestShouldRetry:
    """Tests for should_retry function."""

    def test_should_retry_with_feedback_under_limit(self):
        """Test retry when revision feedback exists and under retry limit."""
        state = AuthorState(
            scenario_config={},
            author_rating=2,
            retry_count=1,
            revision_feedback="Add more detail",
        )

        result = should_retry(state)

        assert result == "generate"

    def test_should_not_retry_without_feedback(self):
        """Test no retry when no revision feedback exists."""
        state = AuthorState(
            scenario_config={},
            author_rating=4,
            retry_count=0,
            revision_feedback=None,
        )

        result = should_retry(state)

        assert result == "save"

    def test_should_not_retry_at_limit(self):
        """Test no retry when at retry limit even with feedback."""
        state = AuthorState(
            scenario_config={},
            author_rating=2,
            retry_count=MAX_RETRY_ATTEMPTS,
            revision_feedback="Still needs work",
        )

        result = should_retry(state)

        assert result == "save"

    def test_should_not_retry_no_feedback(self):
        """Test no retry when no revision feedback provided."""
        state = AuthorState(
            scenario_config={},
            author_rating=None,
            retry_count=0,
            revision_feedback=None,
        )

        result = should_retry(state)

        assert result == "save"

    def test_retry_with_feedback(self):
        """Test retry when revision feedback is present."""
        # With feedback and under limit should generate
        state_with_feedback = AuthorState(
            scenario_config={},
            author_rating=ACCEPTABLE_RATING_THRESHOLD,
            retry_count=0,
            revision_feedback="Please revise",
        )

        result = should_retry(state_with_feedback)
        assert result == "generate"

        # Without feedback should save regardless of rating
        state_without_feedback = AuthorState(
            scenario_config={},
            author_rating=ACCEPTABLE_RATING_THRESHOLD - 1,
            retry_count=0,
            revision_feedback=None,
        )

        result = should_retry(state_without_feedback)
        assert result == "save"


class TestSaveScenario:
    """Tests for save_scenario function."""

    @pytest.mark.asyncio
    async def test_save_scenario_success(self):
        """Test successful scenario save."""
        scenario_draft = ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Fracture",
            learning_objectives=["Learn", "Practice skill"],
            initial_narrative="Start",
            hidden_state="Hidden",
            scene_state="Scene",
        )

        state = AuthorState(
            scenario_config={},
            scenario_draft=scenario_draft.model_dump(),
            scenario_id="scn-save-123",
        )

        mock_store = AsyncMock()
        mock_store.aput.return_value = None

        with patch(
            "summit_sim.graphs.utils.get_scenario_store",
            return_value=mock_store,
        ):
            result = await save_scenario(state)

            mock_store.aput.assert_called_once_with(
                ("scenarios",),
                "scn-save-123",
                {"scenario_draft": scenario_draft.model_dump()},
            )
            assert result == {}

    @pytest.mark.asyncio
    async def test_save_scenario_no_draft(self):
        """Test save when no draft available."""
        state = AuthorState(
            scenario_config={},
            scenario_id="scn-empty",
            scenario_draft=None,
        )

        mock_store = AsyncMock()

        with patch(
            "summit_sim.graphs.utils.get_scenario_store",
            return_value=mock_store,
        ):
            result = await save_scenario(state)

            # Should not call aput when no draft
            mock_store.aput.assert_not_called()
            assert result == {}

    @pytest.mark.asyncio
    async def test_save_scenario_no_id(self):
        """Test save when no scenario ID."""
        scenario_draft = ScenarioDraft(
            title="Test",
            setting="Test",
            patient_summary="Test",
            hidden_truth="Test",
            learning_objectives=["Test", "Validate approach"],
            initial_narrative="Test",
            hidden_state="Test",
            scene_state="Test",
        )

        state = AuthorState(
            scenario_config={},
            scenario_draft=scenario_draft.model_dump(),
            scenario_id="",
        )

        mock_store = AsyncMock()

        with patch(
            "summit_sim.graphs.utils.get_scenario_store",
            return_value=mock_store,
        ):
            result = await save_scenario(state)

            # Should not call aput when no ID
            mock_store.aput.assert_not_called()
            assert result == {}


class TestPresentForAuthor:
    """Tests for present_for_author function."""

    def test_present_scenario_for_review(self):
        """Test presenting scenario for author review."""
        scenario_draft = ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Fracture",
            learning_objectives=["Learn", "Practice skill"],
            initial_narrative="Start",
            hidden_state="Hidden",
            scene_state="Scene",
        )

        state = AuthorState(
            scenario_config={},
            scenario_draft=scenario_draft.model_dump(),
            scenario_id="scn-review-123",
            retry_count=1,
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "approve", "rating": 4}

            result = present_for_author(state)

            # Verify interrupt was called
            call_args = mock_interrupt.call_args[0][0]
            assert call_args["type"] == "scenario_review"
            assert call_args["scenario_id"] == "scn-review-123"
            assert call_args["retry_count"] == 1

            # Verify result
            assert result["approval_status"] == "approved"
            assert result["revision_feedback"] is None

    def test_present_rejected_scenario(self):
        """Test presenting scenario that gets rejected."""
        scenario_draft = ScenarioDraft(
            title="Test",
            setting="Test",
            patient_summary="Test",
            hidden_truth="Test",
            learning_objectives=["Test", "Validate approach"],
            initial_narrative="Test",
            hidden_state="Test",
            scene_state="Test",
        )

        state = AuthorState(
            scenario_config={},
            scenario_draft=scenario_draft.model_dump(),
            scenario_id="scn-reject",
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {
                "action": "revise",
                "feedback": "Needs more detail",
            }

            result = present_for_author(state)

            assert result["approval_status"] == "revision_requested"
            assert result["revision_feedback"] == "Needs more detail"

    def test_present_invalid_action_raises(self):
        """Test that invalid action raises ValueError."""
        scenario_draft = ScenarioDraft(
            title="Test",
            setting="Test",
            patient_summary="Test",
            hidden_truth="Test",
            learning_objectives=["Test", "Validate approach"],
            initial_narrative="Test",
            hidden_state="Test",
            scene_state="Test",
        )

        state = AuthorState(
            scenario_config={},
            scenario_draft=scenario_draft.model_dump(),
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            # Test invalid action
            mock_interrupt.return_value = {"action": "invalid", "rating": 4}

            with pytest.raises(ValueError, match="Invalid action"):
                present_for_author(state)

            # Test missing action
            mock_interrupt.return_value = {"rating": 4}

            with pytest.raises(ValueError, match="Invalid action"):
                present_for_author(state)

    def test_present_no_scenario_raises(self):
        """Test that missing scenario raises ValueError."""
        state = AuthorState(
            scenario_config={},
            scenario_draft=None,
        )

        with pytest.raises(ValueError, match="No scenario draft available"):
            present_for_author(state)


class TestCreateAuthorGraph:
    """Tests for create_author_graph function."""

    def test_create_graph(self):
        """Test that author graph can be created."""
        # Create a mock checkpointer that passes LangGraph's type validation
        mock_checkpointer = AsyncMock(spec=BaseCheckpointSaver)

        graph = create_author_graph(checkpointer=mock_checkpointer)

        assert graph is not None

    def test_create_graph_basic(self):
        """Test that graph can be created successfully."""
        # Just verify graph creation works - LangGraph validates checkpointer types
        mock_checkpointer = AsyncMock(spec=BaseCheckpointSaver)

        graph = create_author_graph(checkpointer=mock_checkpointer)

        assert graph is not None


class TestConstants:
    """Tests for module constants."""

    def test_rating_constants(self):
        """Test rating-related constants."""
        assert MIN_RATING == 1
        assert MAX_RATING == 5
        assert ACCEPTABLE_RATING_THRESHOLD == 3
        assert MAX_RETRY_ATTEMPTS == 3
        assert MIN_RATING < ACCEPTABLE_RATING_THRESHOLD < MAX_RATING
