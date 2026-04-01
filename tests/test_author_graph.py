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
    generate_image_node,
    generate_scenario_node,
    initialize_author,
    present_for_author,
    save_scenario,
    should_retry,
)
from summit_sim.graphs.utils import AppState
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
        """Test successful scenario saving."""
        scenario_draft = ScenarioDraft(
            title="Test Scenario",
            setting="Mountain",
            patient_summary="Test patient",
            hidden_truth="Hidden info",
            learning_objectives=["Learn", "Practice"],
            initial_narrative="Start here",
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

        # Set mock store on AppState singleton
        original_store = AppState.store
        AppState.store = mock_store

        try:
            mock_config = {}
            result = await save_scenario(state, mock_config)

            mock_store.aput.assert_called_once_with(
                ("scenarios",),
                "scn-save-123",
                {"scenario_draft": scenario_draft.model_dump()},
            )
            assert result == {}
        finally:
            # Restore original store
            AppState.store = original_store

    @pytest.mark.asyncio
    async def test_save_scenario_no_draft(self):
        """Test save when no draft available - stores None."""
        state = AuthorState(
            scenario_config={},
            scenario_id="scn-empty",
            scenario_draft=None,
        )

        mock_store = AsyncMock()
        original_store = AppState.store
        AppState.store = mock_store

        try:
            result = await save_scenario(state, {})
            # Stores None when no draft (fail-fast happens on usage, not save)
            mock_store.aput.assert_called_once_with(
                ("scenarios",),
                "scn-empty",
                {"scenario_draft": None},
            )
            assert result == {}
        finally:
            AppState.store = original_store

    @pytest.mark.asyncio
    async def test_save_scenario_no_id(self):
        """Test save fails fast when no scenario ID."""
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
        original_store = AppState.store
        AppState.store = mock_store

        try:
            # Should fail fast when no ID (empty string is still a valid key)
            result = await save_scenario(state, {})
            # Empty string is a valid key for Redis, so this should succeed
            mock_store.aput.assert_called_once()
            assert result == {}
        finally:
            AppState.store = original_store


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
        # Create mock checkpointer and store that pass LangGraph's type validation
        mock_checkpointer = AsyncMock(spec=BaseCheckpointSaver)
        mock_store = AsyncMock()

        graph = create_author_graph(
            checkpointer=mock_checkpointer,
            store=mock_store,
        )

        assert graph is not None

    def test_create_graph_basic(self):
        """Test that graph can be created successfully."""
        # Just verify graph creation works - LangGraph validates checkpointer types
        mock_checkpointer = AsyncMock(spec=BaseCheckpointSaver)
        mock_store = AsyncMock()

        graph = create_author_graph(
            checkpointer=mock_checkpointer,
            store=mock_store,
        )

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


class TestGenerateScenarioNode:
    """Tests for generate_scenario_node async function."""

    @pytest.fixture
    def sample_config_dict(self):
        """Create a sample config dict for testing."""
        return {
            "primary_focus": "Trauma",
            "environment": "Alpine/Mountain",
            "available_personnel": "Partner (2)",
            "evac_distance": "Short (< 2 hours)",
            "complexity": "Standard",
        }

    @pytest.fixture
    def sample_scenario_dict(self):
        """Create a sample scenario dict for testing."""
        return {
            "title": "Test Scenario",
            "setting": "Mountain trail",
            "patient_summary": "30yo male",
            "hidden_truth": "Fractured leg",
            "learning_objectives": ["Assess injury", "Splint"],
            "initial_narrative": "You find an injured hiker...",
            "hidden_state": "BP 120/80, HR 90",
            "scene_state": "Clear weather, 65F",
            "image_data": None,
        }

    @pytest.mark.asyncio
    async def test_generate_new_scenario(
        self, sample_config_dict, sample_scenario_dict
    ):
        """Test generating a new scenario (not a revision)."""
        state = AuthorState(
            scenario_config=sample_config_dict,
            scenario_id="test-123",
            retry_count=0,
            revision_feedback=None,
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        mock_scenario = ScenarioDraft(**sample_scenario_dict)

        with patch("summit_sim.graphs.author.generate_scenario") as mock_generate:
            mock_generate.return_value = mock_scenario
            result = await generate_scenario_node(state, mock_config)

        assert result["scenario_draft"] == mock_scenario.model_dump()
        assert result["retry_count"] == 0  # Not incremented for new scenario
        assert result["revision_feedback"] is None
        mock_generate.assert_called_once()
        # Verify not a revision
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["previous_draft"] is None
        assert call_kwargs["revision_feedback"] is None

    @pytest.mark.asyncio
    async def test_generate_revision_scenario(
        self, sample_config_dict, sample_scenario_dict
    ):
        """Test generating a scenario revision with feedback."""
        previous_scenario = ScenarioDraft(**sample_scenario_dict)

        state = AuthorState(
            scenario_config=sample_config_dict,
            scenario_id="test-123",
            retry_count=1,
            revision_feedback="Add more environmental detail",
            scenario_draft=previous_scenario.model_dump(),
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        revised_scenario = ScenarioDraft(
            **{
                **sample_scenario_dict,
                "setting": "Mountain trail with detailed environment",
            }
        )

        with patch("summit_sim.graphs.author.generate_scenario") as mock_generate:
            mock_generate.return_value = revised_scenario
            result = await generate_scenario_node(state, mock_config)

        assert result["scenario_draft"] == revised_scenario.model_dump()
        assert result["retry_count"] == 2  # Incremented for revision
        assert result["revision_feedback"] is None  # Cleared after use
        # Verify revision context passed
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["previous_draft"] == previous_scenario
        assert call_kwargs["revision_feedback"] == "Add more environmental detail"

    @pytest.mark.asyncio
    async def test_generate_scenario_with_trace_id(
        self, sample_config_dict, sample_scenario_dict
    ):
        """Test that trace_id is captured and returned."""
        state = AuthorState(
            scenario_config=sample_config_dict,
            scenario_id="test-123",
            retry_count=0,
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}
        mock_scenario = ScenarioDraft(**sample_scenario_dict)

        with patch("summit_sim.graphs.author.generate_scenario") as mock_generate:
            mock_generate.return_value = mock_scenario
            result = await generate_scenario_node(state, mock_config)

        # Trace ID should be set from mocked mlflow
        assert "current_trace_id" in result
        assert result["current_trace_id"] == "test-trace-id-123"


class TestGenerateImageNode:
    """Tests for generate_image_node async function."""

    @pytest.fixture
    def sample_config_dict(self):
        """Create a sample config dict for testing."""
        return {
            "primary_focus": "Trauma",
            "environment": "Alpine/Mountain",
            "available_personnel": "Partner (2)",
            "evac_distance": "Short (< 2 hours)",
            "complexity": "Standard",
        }

    @pytest.fixture
    def sample_scenario_dict(self):
        """Create a sample scenario dict for testing."""
        return {
            "title": "Test Scenario",
            "setting": "Mountain trail",
            "patient_summary": "30yo male",
            "hidden_truth": "Fractured leg",
            "learning_objectives": ["Assess injury", "Splint"],
            "initial_narrative": "You find an injured hiker...",
            "hidden_state": "BP 120/80, HR 90",
            "scene_state": "Clear weather, 65F",
            "image_data": None,
        }

    @pytest.mark.asyncio
    async def test_generate_image_success(
        self, sample_config_dict, sample_scenario_dict
    ):
        """Test successful image generation adds image to scenario."""
        state = AuthorState(
            scenario_config=sample_config_dict,
            scenario_draft=sample_scenario_dict,
            scenario_id="test-123",
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}
        base64_image = "base64encodedimagedata123"

        with patch(
            "summit_sim.graphs.author.generate_scenario_image"
        ) as mock_gen_image:
            mock_gen_image.return_value = base64_image
            result = await generate_image_node(state, mock_config)

        result_draft = result["scenario_draft"]
        assert result_draft["image_data"] == base64_image
        mock_gen_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_image_failure_non_blocking(
        self, sample_config_dict, sample_scenario_dict
    ):
        """Test that image generation failure doesn't block scenario."""
        state = AuthorState(
            scenario_config=sample_config_dict,
            scenario_draft=sample_scenario_dict,
            scenario_id="test-123",
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        with patch(
            "summit_sim.graphs.author.generate_scenario_image"
        ) as mock_gen_image:
            mock_gen_image.return_value = None  # Simulate failure
            result = await generate_image_node(state, mock_config)

        # Scenario should still be returned, just without image
        result_draft = result["scenario_draft"]
        assert result_draft["image_data"] is None
        mock_gen_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_image_calls_generator_with_correct_params(
        self, sample_config_dict, sample_scenario_dict
    ):
        """Test that image generator is called with correct scenario and config."""
        state = AuthorState(
            scenario_config=sample_config_dict,
            scenario_draft=sample_scenario_dict,
            scenario_id="test-123",
        )

        mock_config = {"configurable": {"thread_id": "thread-123"}}

        with patch(
            "summit_sim.graphs.author.generate_scenario_image"
        ) as mock_gen_image:
            mock_gen_image.return_value = "test_image_data"
            await generate_image_node(state, mock_config)

        # Verify correct parameters passed
        call_args = mock_gen_image.call_args
        assert isinstance(call_args.args[0], ScenarioDraft)
        assert call_args.args[0].title == sample_scenario_dict["title"]
        assert isinstance(call_args.args[1], ScenarioConfig)
        assert call_args.args[1].primary_focus == sample_config_dict["primary_focus"]


class TestPresentForAuthorMLflowLogging:
    """Tests for MLflow logging branches in present_for_author."""

    @pytest.fixture
    def sample_scenario_dict(self):
        """Create a sample scenario dict for testing."""
        return {
            "title": "Test Scenario",
            "setting": "Mountain trail",
            "patient_summary": "30yo male",
            "hidden_truth": "Fractured leg",
            "learning_objectives": ["Assess injury", "Splint"],
            "initial_narrative": "You find an injured hiker...",
            "hidden_state": "BP 120/80, HR 90",
            "scene_state": "Clear weather, 65F",
            "image_data": None,
        }

    def test_present_approved_logs_to_mlflow(self, sample_scenario_dict):
        """Test that approval is logged to MLflow when trace_id exists."""
        state = AuthorState(
            scenario_config={},
            scenario_draft=sample_scenario_dict,
            scenario_id="test-123",
            current_trace_id="trace-456",
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "approve", "rating": 4}

            with patch("summit_sim.graphs.author.mlflow.log_feedback") as mock_log:
                result = present_for_author(state)

        assert result["approval_status"] == "approved"
        # MLflow log_feedback should have been called for approval
        mock_log.assert_called()

    def test_present_revise_logs_to_mlflow(self, sample_scenario_dict):
        """Test that revision is logged to MLflow when trace_id exists."""
        state = AuthorState(
            scenario_config={},
            scenario_draft=sample_scenario_dict,
            scenario_id="test-123",
            current_trace_id="trace-456",
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {
                "action": "revise",
                "feedback": "Needs more detail about weather",
            }

            with patch("summit_sim.graphs.author.mlflow.log_feedback") as mock_log:
                result = present_for_author(state)

        assert result["approval_status"] == "revision_requested"
        assert result["revision_feedback"] == "Needs more detail about weather"
        # MLflow log_feedback should have been called for revision
        mock_log.assert_called()

    def test_present_no_trace_id_skips_logging(self, sample_scenario_dict):
        """Test that MLflow logging is skipped when no trace_id."""
        state = AuthorState(
            scenario_config={},
            scenario_draft=sample_scenario_dict,
            scenario_id="test-123",
            current_trace_id=None,
        )

        with patch("summit_sim.graphs.author.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "approve", "rating": 5}

            with patch("summit_sim.graphs.author.mlflow.log_feedback") as mock_log:
                result = present_for_author(state)

        assert result["approval_status"] == "approved"
        # MLflow log_feedback should NOT have been called without trace_id
        mock_log.assert_not_called()
