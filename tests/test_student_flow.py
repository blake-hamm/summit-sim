"""Tests for the student flow - simulation graph and storage."""

import pytest

from summit_sim.graphs.simulation import (
    SimulationState,
    create_simulation_graph,
)
from summit_sim.schemas import (
    ChoiceOption,
    ScenarioDraft,
    ScenarioTurn,
)


@pytest.fixture
def sample_scenario():
    """Create a sample scenario with 3 turns for testing."""
    return ScenarioDraft(
        title="Test Emergency",
        setting="Mountain trail at 8000ft",
        patient_summary="30yo male, unconscious",
        hidden_truth="Severe dehydration",
        learning_objectives=["Assess ABCs", "Call for help", "Treat for shock"],
        turns=[
            ScenarioTurn(
                turn_id=0,
                narrative_text="You find a hiker lying on the trail.",
                scene_state={"weather": "clear", "temperature": "60F"},
                choices=[
                    ChoiceOption(
                        choice_id="check_airway",
                        description="Check airway and breathing",
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
            ),
            ScenarioTurn(
                turn_id=1,
                narrative_text="The hiker has a weak pulse but is breathing.",
                scene_state={"weather": "clear", "temperature": "60F"},
                choices=[
                    ChoiceOption(
                        choice_id="call_911",
                        description="Call 911 for evacuation",
                        is_correct=True,
                        next_turn_id=2,
                    ),
                    ChoiceOption(
                        choice_id="wait",
                        description="Wait and see if he improves",
                        is_correct=False,
                        next_turn_id=2,
                    ),
                ],
            ),
            ScenarioTurn(
                turn_id=2,
                narrative_text="Helicopter is en route. Patient remains unconscious.",
                scene_state={"weather": "clear", "temperature": "60F"},
                choices=[
                    ChoiceOption(
                        choice_id="monitor",
                        description="Monitor vitals and keep warm",
                        is_correct=True,
                        next_turn_id=None,
                    ),
                    ChoiceOption(
                        choice_id="move",
                        description="Try to move patient to better location",
                        is_correct=False,
                        next_turn_id=None,
                    ),
                ],
            ),
        ],
        starting_turn_id=0,
    )


class TestSimulationState:
    """Tests for SimulationState."""

    def test_simulation_state_creation(self, sample_scenario):
        """Test creating simulation state."""
        state = SimulationState(
            scenario_draft=sample_scenario.model_dump(),
            current_turn_id=0,
            transcript=[],
            is_complete=False,
        )

        assert state.current_turn_id == 0
        assert state.is_complete is False
        assert state.transcript == []

    def test_from_graph_result(self, sample_scenario):
        """Test creating state from graph result."""
        result = {
            "scenario_draft": sample_scenario.model_dump(),
            "current_turn_id": 1,
            "transcript": [],
            "is_complete": False,
            "key_learning_moments": [],
            "last_selected_choice": None,
            "simulation_result": None,
            "scenario_id": "scn-123",
            "class_id": "abc123",
            "debrief_report": None,
        }

        state = SimulationState.from_graph_result(result)
        assert state.current_turn_id == 1
        assert state.scenario_id == "scn-123"


class TestSimulationGraph:
    """Tests for simulation graph creation and nodes."""

    def test_create_simulation_graph(self):
        """Test that simulation graph can be created."""
        graph = create_simulation_graph()
        assert graph is not None
