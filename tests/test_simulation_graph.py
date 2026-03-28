"""Tests for the simulation graph workflow."""

from typing import Any
from unittest.mock import patch

import pytest

from summit_sim.agents import utils as agent_utils
from summit_sim.graphs.simulation import (
    SimulationState,
    TranscriptEntry,
    check_simulation_completion,
    create_simulation_graph,
    initialize_simulation,
    present_turn,
    update_simulation_state,
)
from summit_sim.schemas import (
    ChoiceOption,
    DebriefReport,
    ScenarioDraft,
    ScenarioTurn,
    SimulationResult,
)


@pytest.fixture
def sample_scenario():
    """Create a sample 3-turn scenario for testing."""
    turn0 = ScenarioTurn(
        turn_id=0,
        narrative_text="Patient is unconscious.",
        choices=[
            ChoiceOption(
                choice_id="check_airway",
                description="Check airway and breathing",
                is_correct=True,
                next_turn_id=1,
            ),
            ChoiceOption(
                choice_id="shake",
                description="Shake patient to wake them",
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
    )

    turn1 = ScenarioTurn(
        turn_id=1,
        narrative_text="Patient is breathing but unresponsive.",
        choices=[
            ChoiceOption(
                choice_id="call_help",
                description="Call for emergency help",
                is_correct=True,
                next_turn_id=2,
            ),
            ChoiceOption(
                choice_id="wait",
                description="Wait to see if they wake up",
                is_correct=False,
                next_turn_id=2,
            ),
            ChoiceOption(
                choice_id="panic",
                description="Panic",
                is_correct=False,
                next_turn_id=2,
            ),
        ],
    )

    turn2 = ScenarioTurn(
        turn_id=2,
        narrative_text="Help is on the way. Patient stable.",
        choices=[
            ChoiceOption(
                choice_id="monitor",
                description="Monitor vital signs",
                is_correct=True,
                next_turn_id=None,
            ),
            ChoiceOption(
                choice_id="check_medication",
                description="Check patient medication",
                is_correct=False,
                next_turn_id=None,
            ),
            ChoiceOption(
                choice_id="panic",
                description="Panic",
                is_correct=False,
                next_turn_id=None,
            ),
        ],
    )

    return ScenarioDraft(
        title="Test Emergency",
        setting="Mountain trail",
        patient_summary="30yo unconscious hiker",
        hidden_truth="Severe dehydration",
        learning_objectives=["Assess ABCs", "Call for help"],
        turns=[turn0, turn1, turn2],
    )


@pytest.fixture
def initial_state(sample_scenario):
    """Create initial state for testing."""
    return SimulationState(
        scenario_draft=sample_scenario.model_dump(),
        current_turn_id=sample_scenario.get_starting_turn().turn_id,
        transcript=[],
        is_complete=False,
        key_learning_moments=[],
        last_selected_choice=None,
        simulation_result=None,
        scenario_id="test-scenario-123",
        debrief_report=None,
    )


def create_test_state(sample_scenario, **overrides):
    """Create a test state with required fields."""
    base = SimulationState(
        scenario_draft=sample_scenario.model_dump(),
        current_turn_id=sample_scenario.get_starting_turn().turn_id,
        transcript=[],
        is_complete=False,
        key_learning_moments=[],
        last_selected_choice=None,
        simulation_result=None,
        scenario_id="test-scenario-123",
        debrief_report=None,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TestInitializeState:
    """Tests for initialize_simulation node."""

    def test_initialize_simulation_valid(self, initial_state, sample_scenario):
        """Test initialization with valid starting turn."""
        result = initialize_simulation(initial_state)

        assert result.scenario_draft == sample_scenario.model_dump()
        assert result.current_turn_id == 0

    def test_initialize_simulation_invalid_turn(self, sample_scenario):
        """Test initialization with invalid starting turn raises error."""
        state = SimulationState(
            scenario_draft=sample_scenario.model_dump(),
            current_turn_id=999,
            transcript=[],
            is_complete=False,
            key_learning_moments=[],
            last_selected_choice=None,
            simulation_result=None,
            scenario_id="test-scenario-123",
            debrief_report=None,
        )

        with pytest.raises(ValueError, match="Starting turn 999 not found"):
            initialize_simulation(state)


class TestPresentTurn:
    """Tests for present_turn node."""

    def test_present_turn_valid(self, initial_state):
        """Test presenting a valid turn."""
        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"choice_id": "check_airway"}
            result = present_turn(initial_state)

        assert result["last_selected_choice"] is not None
        choice_dict = result["last_selected_choice"]
        assert choice_dict["choice_id"] == "check_airway"
        assert choice_dict["is_correct"] is True

    def test_present_turn_invalid_choice(self, initial_state):
        """Test presenting turn with invalid choice raises error."""
        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"choice_id": "invalid_choice"}
            with pytest.raises(ValueError, match="Invalid choice_id"):
                present_turn(initial_state)


class TestUpdateState:
    """Tests for update_simulation_state node."""

    def test_update_simulation_state_appends_transcript(self, initial_state):
        """Test that update_simulation_state appends to transcript."""
        selected_choice = ChoiceOption(
            choice_id="test_choice",
            description="Test description",
            is_correct=True,
            next_turn_id=1,
        )

        result = SimulationResult(
            selected_choice=selected_choice,
            feedback="Good choice!",
            learning_moments=["Learn this"],
            next_turn=None,
            is_complete=False,
        )

        state = SimulationState(
            scenario_draft=initial_state.scenario_draft,
            current_turn_id=initial_state.current_turn_id,
            transcript=[],
            is_complete=False,
            key_learning_moments=[],
            last_selected_choice=selected_choice.model_dump(),
            simulation_result=result.model_dump(),
            scenario_id="test-scenario-123",
            debrief_report=None,
        )

        updated = update_simulation_state(state)

        assert len(updated["transcript"]) == 1
        entry = updated["transcript"][0]
        assert isinstance(entry, TranscriptEntry)
        assert entry.turn_id == 0
        assert entry.choice_id == "test_choice"
        assert entry.feedback == "Good choice!"
        assert entry.learning_moments == ["Learn this"]
        assert entry.next_turn_id == 1

    def test_update_simulation_state_advances_turn(self, initial_state):
        """Test that update_simulation_state advances to next turn."""
        selected_choice = ChoiceOption(
            choice_id="test",
            description="Test",
            is_correct=True,
            next_turn_id=1,
        )

        result = SimulationResult(
            selected_choice=selected_choice,
            feedback="Test",
            learning_moments=[],
            next_turn=None,
            is_complete=False,
        )

        state = SimulationState(
            scenario_draft=initial_state.scenario_draft,
            current_turn_id=initial_state.current_turn_id,
            transcript=[],
            is_complete=False,
            key_learning_moments=[],
            last_selected_choice=selected_choice.model_dump(),
            simulation_result=result.model_dump(),
            scenario_id="test-scenario-123",
            debrief_report=None,
        )

        updated = update_simulation_state(state)

        assert updated["current_turn_id"] == 1
        assert updated["is_complete"] is False

    def test_update_simulation_state_completes_scenario(self, initial_state):
        """Test update_simulation_state marks complete when next_turn_id is None."""
        selected_choice = ChoiceOption(
            choice_id="final_choice",
            description="Final",
            is_correct=True,
            next_turn_id=None,
        )

        result = SimulationResult(
            selected_choice=selected_choice,
            feedback="Scenario complete!",
            learning_moments=["Final lesson"],
            next_turn=None,
            is_complete=True,
        )

        state = SimulationState(
            scenario_draft=initial_state.scenario_draft,
            current_turn_id=initial_state.current_turn_id,
            transcript=[],
            is_complete=False,
            key_learning_moments=[],
            last_selected_choice=selected_choice.model_dump(),
            simulation_result=result.model_dump(),
            scenario_id="test-scenario-123",
            debrief_report=None,
        )

        updated = update_simulation_state(state)

        assert updated["is_complete"] is True
        assert updated["current_turn_id"] == 0

    def test_update_simulation_state_returns_learning_moments(self, initial_state):
        """Test that update_simulation_state returns learning moments from result."""
        selected_choice = ChoiceOption(
            choice_id="test",
            description="Test",
            is_correct=True,
            next_turn_id=1,
        )

        result = SimulationResult(
            selected_choice=selected_choice,
            feedback="Test",
            learning_moments=["New lesson"],
            next_turn=None,
            is_complete=False,
        )

        state = SimulationState(
            scenario_draft=initial_state.scenario_draft,
            current_turn_id=initial_state.current_turn_id,
            transcript=[],
            is_complete=False,
            key_learning_moments=[],
            last_selected_choice=selected_choice.model_dump(),
            simulation_result=result.model_dump(),
            scenario_id="test-scenario-123",
            debrief_report=None,
        )

        updated = update_simulation_state(state)

        assert updated["key_learning_moments"] == ["New lesson"]


class TestCheckCompletion:
    """Tests for check_simulation_completion routing."""

    def test_check_simulation_completion_returns_generate_debrief_when_complete(
        self, initial_state
    ):
        """Test routing to generate_debrief when complete."""
        state = SimulationState(
            scenario_draft=initial_state.scenario_draft,
            current_turn_id=initial_state.current_turn_id,
            transcript=initial_state.transcript,
            is_complete=True,
            key_learning_moments=initial_state.key_learning_moments,
            last_selected_choice=initial_state.last_selected_choice,
            simulation_result=initial_state.simulation_result,
            scenario_id=initial_state.scenario_id,
            debrief_report=initial_state.debrief_report,
        )
        result = check_simulation_completion(state)
        assert result == "generate_debrief"

    def test_check_simulation_completion_returns_present_turn_when_not_complete(
        self, initial_state
    ):
        """Test routing back to present_turn when not complete."""
        state = SimulationState(
            scenario_draft=initial_state.scenario_draft,
            current_turn_id=initial_state.current_turn_id,
            transcript=initial_state.transcript,
            is_complete=False,
            key_learning_moments=initial_state.key_learning_moments,
            last_selected_choice=initial_state.last_selected_choice,
            simulation_result=initial_state.simulation_result,
            scenario_id=initial_state.scenario_id,
            debrief_report=initial_state.debrief_report,
        )
        result = check_simulation_completion(state)
        assert result == "present_turn"


class TestStudentGraphFullCycle:
    """Integration tests for full simulation cycle."""

    @pytest.fixture(autouse=True)
    def mock_api_key(self):
        """Mock the API key to avoid errors during agent creation."""
        with patch(
            "summit_sim.agents.utils.settings.openrouter_api_key", "test-api-key"
        ):
            yield

    @pytest.fixture(autouse=True)
    def clear_agent_cache(self):
        """Clear the agent cache before each test."""
        agent_utils._agent_container.clear()

    @pytest.mark.asyncio
    async def test_full_three_turn_simulation(self, sample_scenario):
        """Test complete 3-turn simulation with mocked agent."""

        async def mock_process_impl(_scenario, current_turn, selected_choice):
            """Mock implementation that returns appropriate result based on turn."""
            if current_turn.turn_id == 0:
                return SimulationResult(
                    selected_choice=selected_choice,
                    feedback="Good job checking the airway!",
                    learning_moments=["Always check ABCs first"],
                    next_turn=sample_scenario.turns[1],
                    is_complete=False,
                )
            if current_turn.turn_id == 1:
                return SimulationResult(
                    selected_choice=selected_choice,
                    feedback="Calling for help was correct!",
                    learning_moments=["Don't hesitate to call for help"],
                    next_turn=sample_scenario.turns[2],
                    is_complete=False,
                )
            return SimulationResult(
                selected_choice=selected_choice,
                feedback="Excellent monitoring!",
                learning_moments=["Continuous monitoring is key"],
                next_turn=None,
                is_complete=True,
            )

        mock_debrief_report = DebriefReport(
            summary="Test summary",
            key_mistakes=[],
            strong_actions=["Good job"],
            best_next_actions=["Practice more"],
            teaching_points=["Key concepts"],
            completion_status="pass",
            final_score=100.0,
        )

        with patch(
            "summit_sim.graphs.simulation.process_choice"
        ) as mock_process_choice:
            mock_process_choice.side_effect = mock_process_impl

            with patch(
                "summit_sim.agents.debrief.generate_debrief"
            ) as mock_generate_debrief:
                mock_generate_debrief.return_value = mock_debrief_report

                with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
                    interrupt_returns = [
                        {"choice_id": "check_airway"},
                        {"choice_id": "call_help"},
                        {"choice_id": "monitor"},
                    ]
                    mock_interrupt.side_effect = interrupt_returns

                    initial_state = SimulationState(
                        scenario_draft=sample_scenario.model_dump(),
                        current_turn_id=0,
                        transcript=[],
                        is_complete=False,
                        key_learning_moments=[],
                        last_selected_choice=None,
                        simulation_result=None,
                        scenario_id="test-scenario-123",
                        debrief_report=None,
                    )

                    graph = create_simulation_graph()
                    config: Any = {"configurable": {"thread_id": "test-thread"}}

                    result = await graph.ainvoke(initial_state, config)
                    state = result

                    for _ in range(2):
                        if state.get("is_complete"):
                            break
                        result = await graph.ainvoke(state, config)
                        state = result

                    assert len(state.get("transcript", [])) == 3
                    assert state.get("is_complete") is True

                    transcript = state.get("transcript", [])
                    assert transcript[0].turn_id == 0
                    assert transcript[0].choice_id == "check_airway"
                    assert transcript[0].feedback == "Good job checking the airway!"

                    assert transcript[1].turn_id == 1
                    assert transcript[1].choice_id == "call_help"
                    assert transcript[1].feedback == "Calling for help was correct!"

                    assert transcript[2].turn_id == 2
                    assert transcript[2].choice_id == "monitor"
                    assert transcript[2].feedback == "Excellent monitoring!"

                    key_moments = state.get("key_learning_moments", [])
                    assert len(key_moments) == 3
                    assert "Continuous monitoring is key" in key_moments

                    assert state.get("debrief_report") is not None

    def test_graph_creation(self):
        """Test that graph can be created successfully."""
        graph = create_simulation_graph()
        assert graph is not None
