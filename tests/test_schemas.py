"""Tests for Pydantic schemas."""

import pytest

from summit_sim.schemas import (
    ChoiceOption,
    HostConfig,
    ScenarioDraft,
    ScenarioTurn,
    SimulationResult,
)


class TestHostConfig:
    """Tests for HostConfig schema."""

    def test_host_config_creation(self):
        """Test creating minimal host configuration."""
        config = HostConfig(
            num_participants=4, activity_type="hiking", difficulty="med"
        )
        assert config.num_participants == 4
        assert config.activity_type == "hiking"
        assert config.difficulty == "med"

    def test_host_config_validation_min(self):
        """Test minimum participant validation."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            HostConfig(num_participants=0, activity_type="skiing", difficulty="low")

    def test_host_config_validation_max(self):
        """Test maximum participant validation."""
        with pytest.raises(ValueError, match="less than or equal to 20"):
            HostConfig(
                num_participants=21, activity_type="canyoneering", difficulty="high"
            )


class TestScenarioTurn:
    """Tests for ScenarioTurn schema."""

    def test_scenario_turn_creation(self):
        """Test creating a scenario turn with choices."""
        choices = [
            ChoiceOption(
                choice_id="choice_a",
                description="Apply tourniquet",
                is_correct=True,
                next_turn_id="turn_2",
            ),
            ChoiceOption(
                choice_id="choice_b",
                description="Apply pressure bandage",
                is_correct=False,
                next_turn_id="turn_2",
            ),
        ]

        turn = ScenarioTurn(
            turn_id="turn_1",
            narrative_text="Patient has severe leg bleeding.",
            choices=choices,
            is_starting_turn=True,
            scene_state={"patient_position": "sitting"},
            hidden_state={"bleeding_rate": "severe"},
        )

        assert turn.turn_id == "turn_1"
        assert len(turn.choices) == 2
        assert turn.is_starting_turn

    def test_scenario_turn_min_choices(self):
        """Test that turns require at least 2 choices."""
        with pytest.raises(ValueError, match="at least 2 items"):
            ScenarioTurn(
                turn_id="turn_1",
                narrative_text="Test",
                choices=[
                    ChoiceOption(
                        choice_id="choice_a",
                        description="Only option",
                        is_correct=True,
                        next_turn_id="turn_2",
                    )
                ],
            )


class TestScenarioDraft:
    """Tests for ScenarioDraft schema."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        choices = [
            ChoiceOption(
                choice_id="treat",
                description="Treat the patient",
                is_correct=True,
                next_turn_id="turn_2",
            ),
            ChoiceOption(
                choice_id="wait",
                description="Wait and observe",
                is_correct=False,
                next_turn_id="turn_2",
            ),
        ]

        turn1 = ScenarioTurn(
            turn_id="turn_1",
            narrative_text="Patient is bleeding.",
            choices=choices,
            is_starting_turn=True,
        )

        turn2 = ScenarioTurn(
            turn_id="turn_2",
            narrative_text="Treatment applied.",
            choices=[
                ChoiceOption(
                    choice_id="monitor",
                    description="Monitor patient",
                    is_correct=True,
                    next_turn_id=None,
                ),
                ChoiceOption(
                    choice_id="evac",
                    description="Evacuate immediately",
                    is_correct=True,
                    next_turn_id=None,
                ),
            ],
        )

        return ScenarioDraft(
            title="Test Scenario",
            setting="Test Location",
            patient_summary="Test Patient",
            hidden_truth="Test Truth",
            learning_objectives=["Objective 1"],
            turns=[turn1, turn2],
            starting_turn_id="turn_1",
        )

    def test_scenario_draft_creation(self, sample_scenario):
        """Test creating a complete scenario draft."""
        assert sample_scenario.title == "Test Scenario"
        assert len(sample_scenario.turns) == 2
        assert sample_scenario.starting_turn_id == "turn_1"

    def test_get_turn(self, sample_scenario):
        """Test retrieving turns by ID."""
        turn = sample_scenario.get_turn("turn_1")
        assert turn is not None
        assert turn.turn_id == "turn_1"

    def test_get_turn_not_found(self, sample_scenario):
        """Test retrieving non-existent turn."""
        turn = sample_scenario.get_turn("nonexistent")
        assert turn is None


class TestSimulationResult:
    """Tests for SimulationResult schema."""

    def test_simulation_result_creation(self):
        """Test creating a simulation result."""
        choice = ChoiceOption(
            choice_id="choice_1",
            description="Test choice",
            is_correct=True,
            next_turn_id=None,
        )

        result = SimulationResult(
            selected_choice=choice,
            feedback="Good work!",
            learning_moments=["Key insight"],
            next_turn=None,
            is_complete=True,
        )

        assert result.selected_choice.choice_id == "choice_1"
        assert result.feedback == "Good work!"
        assert result.is_complete is True


class TestChoiceOption:
    """Tests for ChoiceOption schema."""

    def test_choice_option_creation(self):
        """Test creating a choice option."""
        choice = ChoiceOption(
            choice_id="apply_tourniquet",
            description="Apply a tourniquet to the bleeding leg",
            is_correct=True,
            next_turn_id="turn_2",
        )

        assert choice.choice_id == "apply_tourniquet"
        assert choice.is_correct is True
        assert choice.next_turn_id == "turn_2"

    def test_choice_option_end_scenario(self):
        """Test creating a choice that ends the scenario."""
        choice = ChoiceOption(
            choice_id="evacuate",
            description="Evacuate the patient",
            is_correct=True,
            next_turn_id=None,
        )

        assert choice.next_turn_id is None
