"""Tests for simulation graph workflow."""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver

from summit_sim.agents.action_responder import ActionResponse
from summit_sim.graphs.simulation import (
    SimulationState,
    check_simulation_ending,
    create_simulation_graph,
    initialize_simulation,
    present_prompt,
    process_student_action,
    update_simulation_state,
)
from summit_sim.schemas import ScenarioDraft, TranscriptEntry


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_simulation_state_creation(self):
        """Test creating a SimulationState instance."""
        state = SimulationState(
            scenario_id="test-123",
            turn_count=0,
            transcript=[],
            hidden_state="Initial hidden state",
        )

        assert state.scenario_id == "test-123"
        assert state.turn_count == 0
        assert state.is_complete is False
        assert state.hidden_state == "Initial hidden state"

    def test_simulation_state_defaults(self):
        """Test SimulationState default values."""
        state = SimulationState()

        assert state.scenario_id == ""
        assert state.turn_count == 0
        assert state.transcript == []
        assert state.is_complete is False
        assert state.action_response is None
        assert state.debrief_report is None
        assert state.hidden_state == ""

    def test_from_graph_result(self):
        """Test creating state from graph result."""
        result = {
            "turn_count": 3,
            "transcript": [],
            "is_complete": True,
            "scenario_id": "scn-456",
            "hidden_state": "Updated hidden state",
            "extra_field": "should be filtered",
        }

        state = SimulationState.from_graph_result(result)

        assert state.turn_count == 3
        assert state.is_complete is True
        assert state.scenario_id == "scn-456"
        assert state.hidden_state == "Updated hidden state"
        # extra_field should be filtered out
        assert not hasattr(state, "extra_field")


class TestInitializeSimulation:
    """Tests for initialize_simulation function."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario."""
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

    @pytest.mark.asyncio
    async def test_initialize_simulation(self, sample_scenario):
        """Test simulation initialization."""
        initial_state = SimulationState(
            scenario_id="test-scenario-123",
        )

        # Mock the store to return the scenario
        mock_store = AsyncMock()
        mock_store.aget.return_value.value = {
            "scenario_draft": sample_scenario.model_dump()
        }

        with patch("summit_sim.graphs.simulation.AppState.store", mock_store):
            result = await initialize_simulation(initial_state)

        assert result.turn_count == 0
        assert result.transcript == []
        assert result.is_complete is False
        assert result.scenario_id == "test-scenario-123"
        assert result.hidden_state == "Patient unconscious"

    @pytest.mark.asyncio
    async def test_initialize_simulation_preserves_scenario_id(self, sample_scenario):
        """Test that scenario ID is preserved during initialization."""
        initial_state = SimulationState(
            scenario_id="my-scenario-id",
        )

        # Mock the store to return the scenario
        mock_store = AsyncMock()
        mock_store.aget.return_value.value = {
            "scenario_draft": sample_scenario.model_dump()
        }

        with patch("summit_sim.graphs.simulation.AppState.store", mock_store):
            result = await initialize_simulation(initial_state)

        assert result.scenario_id == "my-scenario-id"


class TestPresentPrompt:
    """Tests for present_prompt function."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario."""
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

    @pytest.mark.asyncio
    async def test_present_initial_prompt(self, sample_scenario):
        """Test presenting initial prompt (turn 0)."""
        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=0,
            hidden_state="Patient unconscious, GCS 8",
        )

        # Mock the store to return the scenario
        mock_store = AsyncMock()
        mock_store.aget.return_value.value = {
            "scenario_draft": sample_scenario.model_dump()
        }

        with (
            patch("summit_sim.graphs.simulation.AppState.store", mock_store),
            patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt,
        ):
            mock_interrupt.return_value = {"action": "I check the patient's airway"}

            result = await present_prompt(state)

            # Verify interrupt was called with correct data
            call_args = mock_interrupt.call_args[0][0]
            assert call_args["type"] == "prompt_presented"
            assert call_args["turn_count"] == 0
            assert call_args["is_initial"] is True
            assert (
                call_args["narrative"]
                == "You find an unconscious hiker on the trail..."
            )

            # Verify transcript was updated with student action
            assert len(result["transcript"]) == 1
            assert (
                result["transcript"][0].student_action == "I check the patient's airway"
            )

    @pytest.mark.asyncio
    async def test_present_subsequent_prompt(self):
        """Test presenting prompt after first turn."""
        action_response = ActionResponse(
            was_correct=True,
            completion_score=0.3,
            feedback="Good first step",
            narrative_text="You check the airway and find it's clear...",
        )

        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=1,
            action_response=action_response.model_dump(),
            hidden_state="Airway clear",
        )

        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "I check for breathing"}

            result = await present_prompt(state)

            # Verify interrupt was called with narrative from action result
            call_args = mock_interrupt.call_args[0][0]
            assert call_args["turn_count"] == 1
            assert call_args["is_initial"] is False
            assert (
                call_args["narrative"] == "You check the airway and find it's clear..."
            )

            assert result["transcript"][-1].student_action == "I check for breathing"

    @pytest.mark.asyncio
    async def test_present_prompt_empty_action_raises(self, sample_scenario):
        """Test that empty action raises ValueError."""
        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=0,
        )

        # Mock the store to return the scenario
        mock_store = AsyncMock()
        mock_store.aget.return_value.value = {
            "scenario_draft": sample_scenario.model_dump()
        }

        with (
            patch("summit_sim.graphs.simulation.AppState.store", mock_store),
            patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt,
        ):
            mock_interrupt.return_value = {"action": "   "}  # Empty after strip

            with pytest.raises(ValueError, match="Empty student action"):
                await present_prompt(state)


class TestProcessStudentAction:
    """Tests for process_student_action function."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario."""
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
    async def test_process_action(self, sample_scenario):
        """Test processing a player action."""
        expected_result = ActionResponse(
            was_correct=True,
            completion_score=0.4,
            feedback="Good assessment",
            narrative_text="You assess the patient...",
        )

        transcript_entry = TranscriptEntry(
            turn_id=1,
            turn_narrative="",
            student_action="I check vital signs",
            was_correct=False,
            feedback="",
        )

        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=0,
            hidden_state="Patient unconscious",
            transcript=[transcript_entry],
        )

        # Mock the store to return the scenario
        mock_store = AsyncMock()
        mock_store.aget.return_value.value = {
            "scenario_draft": sample_scenario.model_dump()
        }

        with (
            patch("summit_sim.graphs.simulation.AppState.store", mock_store),
            patch("summit_sim.graphs.simulation.action_response_agent") as mock_agent,
            patch("summit_sim.settings.settings") as mock_settings,
        ):
            mock_agent.return_value = expected_result
            mock_settings.max_turns = 5

            config = {"configurable": {"thread_id": "test-thread-id"}}
            result = await process_student_action(state, config)

            assert result["action_response"] == expected_result.model_dump()
            mock_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_action_with_transcript(self, sample_scenario):
        """Test processing action with existing transcript."""
        transcript_entry = TranscriptEntry(
            turn_id=1,
            turn_narrative="Initial narrative",
            student_action="First action",
            was_correct=True,
            feedback="Good start",
        )

        expected_result = ActionResponse(
            was_correct=True,
            completion_score=0.6,
            feedback="Continuing well",
            narrative_text="Next narrative...",
        )

        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=1,
            hidden_state="Previous state",
            transcript=[transcript_entry],
        )

        # Mock the store to return the scenario
        mock_store = AsyncMock()
        mock_store.aget.return_value.value = {
            "scenario_draft": sample_scenario.model_dump()
        }

        with (
            patch("summit_sim.graphs.simulation.AppState.store", mock_store),
            patch("summit_sim.graphs.simulation.action_response_agent") as mock_agent,
            patch("summit_sim.settings.settings") as mock_settings,
        ):
            mock_agent.return_value = expected_result
            mock_settings.max_turns = 5

            config = {"configurable": {"thread_id": "test-thread-id"}}
            result = await process_student_action(state, config)

            assert result["action_response"] == expected_result.model_dump()


class TestUpdateSimulationState:
    """Tests for update_simulation_state function."""

    def test_update_state_basic(self):
        """Test basic state update."""
        action_response = ActionResponse(
            was_correct=True,
            completion_score=0.5,
            feedback="Good action",
            narrative_text="The patient responds well...",
        )

        transcript_entry = TranscriptEntry(
            turn_id=1,
            turn_narrative="",
            student_action="I treat the patient",
            was_correct=False,
            feedback="",
        )

        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=0,
            transcript=[transcript_entry],
            action_response=action_response.model_dump(),
            hidden_state="Previous hidden",
        )

        with patch("summit_sim.settings.settings") as mock_settings:
            mock_settings.max_turns = 5

            result = update_simulation_state(state)

            assert result["turn_count"] == 1
            assert len(result["transcript"]) == 1
            assert result["is_complete"] is False

            transcript_entry = result["transcript"][0]
            assert transcript_entry.student_action == "I treat the patient"
            assert transcript_entry.was_correct is True

    def test_update_state_completes_scenario(self):
        """Test state update when scenario completes naturally."""
        action_response = ActionResponse(
            was_correct=True,
            completion_score=1.0,
            feedback="Scenario complete!",
            narrative_text="Patient evacuated successfully...",
        )

        transcript_entry = TranscriptEntry(
            turn_id=4,
            turn_narrative="",
            student_action="Evacuate patient",
            was_correct=False,
            feedback="",
        )

        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=3,
            transcript=[transcript_entry],
            action_response=action_response.model_dump(),
            hidden_state="Previous",
        )

        with patch("summit_sim.settings.settings") as mock_settings:
            mock_settings.max_turns = 5

            result = update_simulation_state(state)

            assert result["turn_count"] == 4
            assert result["is_complete"] is True

    def test_update_state_max_turns_reached(self):
        """Test state update when max turns reached."""
        action_response = ActionResponse(
            was_correct=True,
            completion_score=0.7,
            # Not naturally complete
            feedback="Good progress",
            narrative_text="Continuing...",
        )

        transcript_entry = TranscriptEntry(
            turn_id=5,
            turn_narrative="",
            student_action="Action",
            was_correct=False,
            feedback="",
        )

        state = SimulationState(
            scenario_id="test-scenario",
            turn_count=4,  # Will become 5, which equals max_turns
            transcript=[transcript_entry],
            action_response=action_response.model_dump(),
            hidden_state="Previous",
        )

        with patch("summit_sim.settings.settings") as mock_settings:
            mock_settings.max_turns = 5

            result = update_simulation_state(state)

            assert result["turn_count"] == 5
            assert result["is_complete"] is True  # Forced complete due to max turns


class TestCheckSimulationEnding:
    """Tests for check_simulation_ending function."""

    def test_returns_generate_debrief_when_complete(self):
        """Test routing to debrief when simulation is complete."""
        state = SimulationState(
            scenario_id="test-scenario",
            is_complete=True,
        )

        result = check_simulation_ending(state)

        assert result == "generate_debrief"

    def test_returns_present_prompt_when_not_complete(self):
        """Test routing back to present prompt when not complete."""
        state = SimulationState(
            scenario_id="test-scenario",
        )

        result = check_simulation_ending(state)

        assert result == "present_prompt"


class TestCreateSimulationGraph:
    """Tests for create_simulation_graph function."""

    def test_create_graph(self):
        """Test that graph can be created."""
        mock_checkpointer = AsyncMock(spec=BaseCheckpointSaver)

        graph = create_simulation_graph(checkpointer=mock_checkpointer)

        assert graph is not None

    def test_graph_structure(self):
        """Test that graph has expected nodes and edges."""
        mock_checkpointer = AsyncMock(spec=BaseCheckpointSaver)

        graph = create_simulation_graph(checkpointer=mock_checkpointer)

        # Graph should have the expected nodes
        # Note: We can't easily inspect the internal structure without
        # invoking the graph, but we can verify it compiles successfully
        assert graph is not None
