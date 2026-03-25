"""Tests for the simulation graph workflow."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from summit_sim.agents import config as agent_config
from summit_sim.graphs.simulation import (
    check_completion,
    create_simulation_graph,
    initialize_state,
    present_turn,
    update_state,
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
        ],
        is_starting_turn=True,
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
        ],
    )

    return ScenarioDraft(
        title="Test Emergency",
        setting="Mountain trail",
        patient_summary="30yo unconscious hiker",
        hidden_truth="Severe dehydration",
        learning_objectives=["Assess ABCs", "Call for help"],
        turns=[turn0, turn1, turn2],
        starting_turn_id=0,
    )


@pytest.fixture
def initial_state(sample_scenario):
    """Create initial state for testing."""
    return {
        "scenario_draft": sample_scenario,
        "current_turn_id": sample_scenario.starting_turn_id,
        "transcript": [],
        "is_complete": False,
        "key_learning_moments": [],
        "last_selected_choice": None,
        "simulation_result": None,
        "scenario_id": "test-scenario-123",
        "class_id": None,
        "debrief_report": None,
        "mlflow_run_id": "",
    }


def create_test_state(sample_scenario, **overrides):
    """Create a test state with required fields."""
    base = {
        "scenario_draft": sample_scenario,
        "current_turn_id": sample_scenario.starting_turn_id,
        "transcript": [],
        "is_complete": False,
        "key_learning_moments": [],
        "last_selected_choice": None,
        "simulation_result": None,
        "scenario_id": "test-scenario-123",
        "class_id": None,
        "debrief_report": None,
        "mlflow_run_id": "",
    }
    base.update(overrides)
    return base


class TestInitializeState:
    """Tests for initialize_state node."""

    def test_initialize_state_valid(self, initial_state, sample_scenario):
        """Test initialization with valid starting turn."""
        result = initialize_state(initial_state)

        assert result["scenario_draft"] == sample_scenario
        assert result["current_turn_id"] == 0

    def test_initialize_state_invalid_turn(self, sample_scenario):
        """Test initialization with invalid starting turn raises error."""
        state = {
            "scenario_draft": sample_scenario,
            "current_turn_id": 999,
            "transcript": [],
            "is_complete": False,
            "key_learning_moments": [],
            "last_selected_choice": None,
            "simulation_result": None,
            "scenario_id": "test-scenario-123",
            "class_id": None,
            "debrief_report": None,
            "mlflow_run_id": "",
        }

        with pytest.raises(ValueError, match="Starting turn 999 not found"):
            initialize_state(state)


class TestPresentTurn:
    """Tests for present_turn node."""

    def test_present_turn_valid(self, initial_state):
        """Test presenting a valid turn."""
        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"choice_id": "check_airway"}
            result = present_turn(initial_state)

        assert result["last_selected_choice"] is not None
        assert result["last_selected_choice"].choice_id == "check_airway"
        assert result["last_selected_choice"].is_correct is True

    def test_present_turn_invalid_choice(self, initial_state):
        """Test presenting turn with invalid choice raises error."""
        with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"choice_id": "invalid_choice"}
            with pytest.raises(ValueError, match="Invalid choice_id"):
                present_turn(initial_state)


class TestUpdateState:
    """Tests for update_state node."""

    def test_update_state_appends_transcript(self, initial_state):
        """Test that update_state appends to transcript."""
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

        state = {
            **initial_state,
            "last_selected_choice": selected_choice,
            "simulation_result": result,
        }

        updated = update_state(state)

        assert len(updated["transcript"]) == 1
        entry = updated["transcript"][0]
        assert isinstance(entry, dict)
        assert entry["turn_id"] == 0
        assert entry["choice_id"] == "test_choice"
        assert entry["feedback"] == "Good choice!"
        assert entry["learning_moments"] == ["Learn this"]
        assert entry["next_turn_id"] == 1

    def test_update_state_advances_turn(self, initial_state):
        """Test that update_state advances to next turn."""
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

        state = {
            **initial_state,
            "last_selected_choice": selected_choice,
            "simulation_result": result,
        }

        updated = update_state(state)

        assert updated["current_turn_id"] == 1
        assert updated["is_complete"] is False

    def test_update_state_completes_scenario(self, initial_state):
        """Test that update_state marks scenario complete when next_turn_id is None."""
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

        state = {
            **initial_state,
            "last_selected_choice": selected_choice,
            "simulation_result": result,
        }

        updated = update_state(state)

        assert updated["is_complete"] is True
        assert updated["current_turn_id"] == 0

    def test_update_state_returns_learning_moments(self, initial_state):
        """Test that update_state returns learning moments from result."""
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

        state = {
            **initial_state,
            "last_selected_choice": selected_choice,
            "simulation_result": result,
        }

        updated = update_state(state)

        # update_state returns just the learning moments from the result
        # LangGraph checkpointing handles state persistence
        assert updated["key_learning_moments"] == ["New lesson"]


class TestCheckCompletion:
    """Tests for check_completion routing."""

    def test_check_completion_returns_generate_debrief_when_complete(
        self, initial_state
    ):
        """Test routing to generate_debrief when complete."""
        state = {**initial_state, "is_complete": True}
        result = check_completion(state)
        assert result == "generate_debrief"

    def test_check_completion_returns_present_turn_when_not_complete(
        self, initial_state
    ):
        """Test routing back to present_turn when not complete."""
        state = {**initial_state, "is_complete": False}
        result = check_completion(state)
        assert result == "present_turn"


class TestSimulationGraphFullCycle:
    """Integration tests for full simulation cycle."""

    @pytest.fixture(autouse=True)
    def mock_api_key(self):
        """Mock the API key to avoid errors during agent creation."""
        with patch(
            "summit_sim.agents.config.settings.openrouter_api_key", "test-api-key"
        ):
            yield

    @pytest.fixture(autouse=True)
    def clear_agent_cache(self):
        """Clear the agent cache before each test."""
        agent_config._agent_container.clear()

    @pytest.mark.asyncio
    async def test_full_three_turn_simulation(self, sample_scenario):
        """Test complete 3-turn simulation with mocked agent."""
        call_count = 0

        async def mock_process_impl(_scenario, current_turn, selected_choice):
            """Mock implementation that returns appropriate result based on turn."""
            nonlocal call_count
            call_count += 1
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
                "summit_sim.graphs.simulation.generate_debrief"
            ) as mock_generate_debrief:
                mock_generate_debrief.return_value = mock_debrief_report

                with patch("summit_sim.graphs.simulation.interrupt") as mock_interrupt:
                    with patch(
                        "summit_sim.graphs.simulation.mlflow.start_run"
                    ) as mock_start_run:
                        mock_run = MagicMock()
                        mock_run.info.run_id = "mock-sim-run-id"
                        mock_start_run.return_value.__enter__ = MagicMock(
                            return_value=mock_run
                        )
                        mock_start_run.return_value.__exit__ = MagicMock(
                            return_value=False
                        )

                        interrupt_returns = [
                            {"choice_id": "check_airway"},
                            {"choice_id": "call_help"},
                            {"choice_id": "monitor"},
                        ]
                        mock_interrupt.side_effect = interrupt_returns

                        initial_state = {
                            "scenario_draft": sample_scenario,
                            "current_turn_id": 0,
                            "transcript": [],
                            "is_complete": False,
                            "key_learning_moments": [],
                            "last_selected_choice": None,
                            "simulation_result": None,
                            "scenario_id": "test-scenario-123",
                            "class_id": None,
                            "debrief_report": None,
                            "mlflow_run_id": "",
                        }

                    graph = create_simulation_graph()
                    config: Any = {"configurable": {"thread_id": "test-thread"}}

                    result = await graph.ainvoke(initial_state, config)
                    state = result

                    for _ in range(2):
                        if state["is_complete"]:
                            break
                        result = await graph.ainvoke(state, config)
                        state = result

                    assert len(state["transcript"]) == 3
                    assert state["is_complete"] is True

                    assert state["transcript"][0]["turn_id"] == 0
                    assert state["transcript"][0]["choice_id"] == "check_airway"
                    assert (
                        state["transcript"][0]["feedback"]
                        == "Good job checking the airway!"
                    )

                    assert state["transcript"][1]["turn_id"] == 1
                    assert state["transcript"][1]["choice_id"] == "call_help"
                    assert (
                        state["transcript"][1]["feedback"]
                        == "Calling for help was correct!"
                    )

                    assert state["transcript"][2]["turn_id"] == 2
                    assert state["transcript"][2]["choice_id"] == "monitor"
                    assert state["transcript"][2]["feedback"] == "Excellent monitoring!"

                    # With checkpoint-based state persistence (no append_reducer),
                    # key_learning_moments contains only the current turn's moments
                    assert len(state["key_learning_moments"]) == 3
                    assert (
                        "Continuous monitoring is key" in state["key_learning_moments"]
                    )

                    assert state["debrief_report"] is not None

    def test_graph_creation(self):
        """Test that graph can be created successfully."""
        graph = create_simulation_graph()
        assert graph is not None
