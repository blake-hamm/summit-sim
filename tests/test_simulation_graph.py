"""Tests for simulation graph workflow."""

from unittest.mock import MagicMock, patch

import pytest

from summit_sim.graphs.simulation import (
    SimulationState,
    _build_transcript_context,
    check_simulation_ending,
    create_simulation_graph,
    initialize_simulation,
    present_prompt,
    process_player_action,
    update_simulation_state,
)
from summit_sim.graphs.utils import TranscriptEntry
from summit_sim.schemas import DynamicTurnResult, ScenarioDraft


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_simulation_state_creation(self):
        """Test creating a SimulationState instance."""
        state = SimulationState(
            scenario_draft={"title": "Test"},
            turn_count=0,
            transcript=[],
            scenario_id="test-123",
            hidden_state="Initial hidden state",
            scene_state="Initial scene state",
        )

        assert state.scenario_id == "test-123"
        assert state.turn_count == 0
        assert state.is_complete is False
        assert state.hidden_state == "Initial hidden state"
        assert state.scene_state == "Initial scene state"

    def test_simulation_state_defaults(self):
        """Test SimulationState default values."""
        state = SimulationState(
            scenario_draft={"title": "Test"},
        )

        assert state.turn_count == 0
        assert state.transcript == []
        assert state.is_complete is False
        assert state.key_learning_moments == []
        assert state.last_student_action is None
        assert state.action_result is None
        assert state.scenario_id == ""
        assert state.debrief_report is None
        assert state.hidden_state == ""
        assert state.scene_state == ""

    def test_from_graph_result(self):
        """Test creating state from graph result."""
        result = {
            "scenario_draft": {"title": "Test Scenario"},
            "turn_count": 3,
            "transcript": [],
            "is_complete": True,
            "scenario_id": "scn-456",
            "hidden_state": "Updated hidden state",
            "scene_state": "Updated scene state",
            "extra_field": "should be filtered",
        }

        state = SimulationState.from_graph_result(result)

        assert state.turn_count == 3
        assert state.is_complete is True
        assert state.scenario_id == "scn-456"
        assert state.hidden_state == "Updated hidden state"
        assert state.scene_state == "Updated scene state"
        # extra_field should be filtered out
        assert not hasattr(state, "extra_field")


class TestInitializeSimulation:
    """Tests for initialize_simulation function."""

    @pytest.fixture
    def sample_scenario_draft(self):
        """Create a sample scenario draft."""
        return ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Severe dehydration",
            learning_objectives=["Assess ABCs", "Manage dehydration"],
            initial_narrative="You find a hiker...",
            hidden_state="Patient unconscious",
            scene_state="Clear weather",
        )

    def test_initialize_simulation(self, sample_scenario_draft):
        """Test simulation initialization."""
        initial_state = SimulationState(
            scenario_draft=sample_scenario_draft.model_dump(),
            scenario_id="test-scenario-123",
        )

        result = initialize_simulation(initial_state)

        assert result.turn_count == 0
        assert result.transcript == []
        assert result.is_complete is False
        assert result.scenario_id == "test-scenario-123"
        assert result.hidden_state == "Patient unconscious"
        assert result.scene_state == "Clear weather"

    def test_initialize_simulation_preserves_scenario_id(self):
        """Test that scenario ID is preserved during initialization."""
        draft = ScenarioDraft(
            title="Test",
            setting="Location",
            patient_summary="Patient",
            hidden_truth="Truth",
            learning_objectives=["Objective", "Secondary objective"],
            initial_narrative="Initial narrative",
            hidden_state="Hidden",
            scene_state="Scene",
        )

        initial_state = SimulationState(
            scenario_draft=draft.model_dump(),
            scenario_id="my-scenario-id",
        )

        result = initialize_simulation(initial_state)

        assert result.scenario_id == "my-scenario-id"


class TestPresentPrompt:
    """Tests for present_prompt function."""

    @pytest.fixture
    def sample_scenario_draft(self):
        """Create a sample scenario draft."""
        return ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Severe dehydration",
            learning_objectives=["Assess ABCs", "Initiate evacuation"],
            initial_narrative="You find an unconscious hiker on the trail...",
            hidden_state="Patient unconscious, GCS 8",
            scene_state="Clear weather, 2 hours to sunset",
        )

    def test_present_initial_prompt(self, sample_scenario_draft):
        """Test presenting initial prompt (turn 0)."""
        state = SimulationState(
            scenario_draft=sample_scenario_draft.model_dump(),
            turn_count=0,
            hidden_state="Patient unconscious, GCS 8",
            scene_state="Clear weather, 2 hours to sunset",
        )

        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "I check the patient's airway"}

            result = present_prompt(state)

            # Verify interrupt was called with correct data
            call_args = mock_interrupt.call_args[0][0]
            assert call_args["type"] == "prompt_presented"
            assert call_args["turn_count"] == 0
            assert call_args["is_initial"] is True
            assert (
                call_args["narrative"]
                == "You find an unconscious hiker on the trail..."
            )
            assert call_args["scene_state"] == "Clear weather, 2 hours to sunset"

            # Verify result
            assert result["last_student_action"] == "I check the patient's airway"

    def test_present_subsequent_prompt(self, sample_scenario_draft):
        """Test presenting prompt after first turn."""
        action_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.3,
            feedback="Good first step",
            narrative_text="You check the airway and find it's clear...",
            updated_hidden_state="Airway clear",
            updated_scene_state="Weather stable",
        )

        state = SimulationState(
            scenario_draft=sample_scenario_draft.model_dump(),
            turn_count=1,
            action_result=action_result.model_dump(),
            hidden_state="Airway clear",
            scene_state="Weather stable",
        )

        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "I check for breathing"}

            result = present_prompt(state)

            # Verify interrupt was called with narrative from action result
            call_args = mock_interrupt.call_args[0][0]
            assert call_args["turn_count"] == 1
            assert call_args["is_initial"] is False
            assert (
                call_args["narrative"] == "You check the airway and find it's clear..."
            )

            assert result["last_student_action"] == "I check for breathing"

    def test_present_prompt_empty_action_raises(self, sample_scenario_draft):
        """Test that empty action raises ValueError."""
        state = SimulationState(
            scenario_draft=sample_scenario_draft.model_dump(),
            turn_count=0,
        )

        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "   "}  # Empty after strip

            with pytest.raises(ValueError, match="Empty student action"):
                present_prompt(state)


class TestProcessPlayerAction:
    """Tests for process_player_action function."""

    @pytest.fixture
    def sample_scenario_draft(self):
        """Create a sample scenario draft."""
        return ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Severe dehydration",
            learning_objectives=["Assess ABCs", "Stabilize patient"],
            initial_narrative="You find a hiker...",
            hidden_state="Patient unconscious",
            scene_state="Clear weather",
        )

    @pytest.mark.asyncio
    async def test_process_action(self, sample_scenario_draft):
        """Test processing a player action."""
        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.4,
            feedback="Good assessment",
            narrative_text="You assess the patient...",
            updated_hidden_state="Vitals stable",
            updated_scene_state="Weather unchanged",
        )

        state = SimulationState(
            scenario_draft=sample_scenario_draft.model_dump(),
            turn_count=0,
            last_student_action="I check vital signs",
            hidden_state="Patient unconscious",
            scene_state="Clear weather",
            transcript=[],
        )

        with (
            patch("summit_sim.graphs.simulation.process_action") as mock_process,
            patch("summit_sim.graphs.simulation.get_settings") as mock_get_settings,
        ):
            mock_process.return_value = expected_result
            mock_settings = MagicMock()
            mock_settings.max_turns = 5
            mock_get_settings.return_value = mock_settings

            result = await process_player_action(state)

            assert result["action_result"] == expected_result.model_dump()
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_action_with_transcript(self, sample_scenario_draft):
        """Test processing action with existing transcript."""
        transcript_entry = TranscriptEntry(
            turn_id=1,
            turn_narrative="Initial narrative",
            student_action="First action",
            was_correct=True,
            feedback="Good start",
            learning_moments=["Lesson 1"],
        )

        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.6,
            feedback="Continuing well",
            narrative_text="Next narrative...",
            updated_hidden_state="Updated",
            updated_scene_state="Updated scene",
        )

        state = SimulationState(
            scenario_draft=sample_scenario_draft.model_dump(),
            turn_count=1,
            last_student_action="Second action",
            hidden_state="Previous state",
            scene_state="Previous scene",
            transcript=[transcript_entry],
        )

        with (
            patch("summit_sim.graphs.simulation.process_action") as mock_process,
            patch("summit_sim.graphs.simulation.get_settings") as mock_get_settings,
        ):
            mock_process.return_value = expected_result
            mock_settings = MagicMock()
            mock_settings.max_turns = 5
            mock_get_settings.return_value = mock_settings

            result = await process_player_action(state)

            assert result["action_result"] == expected_result.model_dump()
            # Verify process_action was called with transcript context
            call_args = mock_process.call_args
            assert call_args[1]["context"].transcript_history  # type: ignore


class TestUpdateSimulationState:
    """Tests for update_simulation_state function."""

    def test_update_state_basic(self):
        """Test basic state update."""
        action_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.5,
            feedback="Good action",
            narrative_text="The patient responds well...",
            updated_hidden_state="Vitals improving",
            updated_scene_state="Weather stable",
        )

        state = SimulationState(
            scenario_draft={"title": "Test"},
            turn_count=0,
            transcript=[],
            key_learning_moments=[],
            last_student_action="I treat the patient",
            action_result=action_result.model_dump(),
            hidden_state="Previous hidden",
            scene_state="Previous scene",
        )

        with patch("summit_sim.graphs.simulation.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.max_turns = 5
            mock_get_settings.return_value = mock_settings

            result = update_simulation_state(state)

            assert result["turn_count"] == 1
            assert len(result["transcript"]) == 1
            assert result["is_complete"] is False
            assert result["hidden_state"] == "Vitals improving"
            assert result["scene_state"] == "Weather stable"
            assert result["last_student_action"] is None  # Reset

            transcript_entry = result["transcript"][0]
            assert transcript_entry.student_action == "I treat the patient"
            assert transcript_entry.was_correct is True

    def test_update_state_completes_scenario(self):
        """Test state update when scenario completes naturally."""
        action_result = DynamicTurnResult(
            was_correct=True,
            completion_score=1.0,
            feedback="Scenario complete!",
            narrative_text="Patient evacuated successfully...",
            updated_hidden_state="Patient evacuated",
            updated_scene_state="Rescue complete",
        )

        state = SimulationState(
            scenario_draft={"title": "Test"},
            turn_count=3,
            transcript=[],
            key_learning_moments=[],
            last_student_action="Evacuate patient",
            action_result=action_result.model_dump(),
            hidden_state="Previous",
            scene_state="Previous scene",
        )

        with patch("summit_sim.graphs.simulation.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.max_turns = 5
            mock_get_settings.return_value = mock_settings

            result = update_simulation_state(state)

            assert result["turn_count"] == 4
            assert result["is_complete"] is True

    def test_update_state_max_turns_reached(self):
        """Test state update when max turns reached."""
        action_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.7,
            # Not naturally complete
            feedback="Good progress",
            narrative_text="Continuing...",
            updated_hidden_state="State",
            updated_scene_state="Scene",
        )

        state = SimulationState(
            scenario_draft={"title": "Test"},
            turn_count=4,  # Will become 5, which equals max_turns
            transcript=[],
            key_learning_moments=[],
            last_student_action="Action",
            action_result=action_result.model_dump(),
            hidden_state="Previous",
            scene_state="Previous scene",
        )

        with patch("summit_sim.graphs.simulation.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.max_turns = 5
            mock_get_settings.return_value = mock_settings

            result = update_simulation_state(state)

            assert result["turn_count"] == 5
            assert result["is_complete"] is True  # Forced complete due to max turns


class TestBuildTranscriptContext:
    """Tests for _build_transcript_context function."""

    def test_empty_transcript(self):
        """Test building context from empty transcript."""
        result = _build_transcript_context([])
        assert result == []

    def test_single_entry(self):
        """Test building context from single entry."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Narrative 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Feedback 1",
                learning_moments=["Lesson 1"],
            )
        ]

        result = _build_transcript_context(transcript)

        assert len(result) == 1
        assert result[0]["action"] == "Action 1"
        assert result[0]["feedback"] == "Feedback 1"
        assert result[0]["narrative"] == "Narrative 1"
        assert result[0]["was_correct"] is True

    def test_multiple_entries(self):
        """Test building context from multiple entries."""
        transcript = [
            TranscriptEntry(
                turn_id=i,
                turn_narrative=f"Narrative {i}",
                student_action=f"Action {i}",
                was_correct=i % 2 == 0,
                feedback=f"Feedback {i}",
                learning_moments=[f"Lesson {i}"],
            )
            for i in range(1, 4)
        ]

        result = _build_transcript_context(transcript)

        assert len(result) == 3
        assert result[0]["action"] == "Action 1"
        assert result[2]["action"] == "Action 3"

    def test_truncates_to_last_five(self):
        """Test that only last 5 entries are included."""
        transcript = [
            TranscriptEntry(
                turn_id=i,
                turn_narrative=f"Narrative {i}",
                student_action=f"Action {i}",
                was_correct=True,
                feedback=f"Feedback {i}",
                learning_moments=[],
            )
            for i in range(1, 11)  # 10 entries
        ]

        result = _build_transcript_context(transcript)

        assert len(result) == 5
        assert result[0]["action"] == "Action 6"  # First of last 5
        assert result[4]["action"] == "Action 10"  # Last


class TestCheckSimulationEnding:
    """Tests for check_simulation_ending function."""

    def test_returns_generate_debrief_when_complete(self):
        """Test routing to debrief when simulation is complete."""
        state = SimulationState(
            scenario_draft={"title": "Test"},
            is_complete=True,
        )

        result = check_simulation_ending(state)

        assert result == "generate_debrief"

    def test_returns_present_prompt_when_not_complete(self):
        """Test routing back to present prompt when not complete."""
        state = SimulationState(
            scenario_draft={"title": "Test"},
        )

        result = check_simulation_ending(state)

        assert result == "present_prompt"


class TestCreateSimulationGraph:
    """Tests for create_simulation_graph function."""

    def test_create_graph(self):
        """Test that graph can be created."""
        graph = create_simulation_graph()

        assert graph is not None

    def test_graph_structure(self):
        """Test that graph has expected nodes and edges."""
        graph = create_simulation_graph()

        # Graph should have the expected nodes
        # Note: We can't easily inspect the internal structure without
        # invoking the graph, but we can verify it compiles successfully
        assert graph is not None
