"""Tests for Pydantic schemas."""

import pytest

from summit_sim.schemas import (
    ChoiceOption,
    ScenarioConfig,
    ScenarioDraft,
    ScenarioTurn,
    SimulationResult,
)


class TestScenarioConfig:
    """Tests for ScenarioConfig schema."""

    def test_scenario_config_creation(self):
        """Test creating minimal scenario configuration."""
        config = ScenarioConfig(
            num_participants=4, activity_type="hiking", difficulty="med"
        )
        assert config.num_participants == 4
        assert config.activity_type == "hiking"
        assert config.difficulty == "med"

    def test_scenario_config_validation_min(self):
        """Test minimum participant validation."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            ScenarioConfig(num_participants=0, activity_type="skiing", difficulty="low")

    def test_scenario_config_validation_max(self):
        """Test maximum participant validation."""
        with pytest.raises(ValueError, match="less than or equal to 20"):
            ScenarioConfig(
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
                next_turn_id=2,
            ),
            ChoiceOption(
                choice_id="choice_b",
                description="Apply pressure bandage",
                is_correct=False,
                next_turn_id=2,
            ),
            ChoiceOption(
                choice_id="panic",
                description="Panic",
                is_correct=False,
                next_turn_id=2,
            ),
        ]

        turn = ScenarioTurn(
            turn_id=0,
            narrative_text="Patient has severe leg bleeding.",
            choices=choices,
            scene_state={"patient_position": "sitting"},
            hidden_state={"bleeding_rate": "severe"},
        )

        assert turn.turn_id == 0
        assert len(turn.choices) == 3

    def test_scenario_turn_min_choices(self):
        """Test that turns require at least 3 choices."""
        with pytest.raises(ValueError, match="List should have at least 3 items"):
            ScenarioTurn(
                turn_id=1,
                narrative_text="Test",
                choices=[
                    ChoiceOption(
                        choice_id="choice_a",
                        description="Only option",
                        is_correct=True,
                        next_turn_id=2,
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
                next_turn_id=2,
            ),
            ChoiceOption(
                choice_id="wait",
                description="Wait and observe",
                is_correct=False,
                next_turn_id=2,
            ),
            ChoiceOption(
                choice_id="panic",
                description="Panic",
                is_correct=False,
                next_turn_id=2,
            ),
        ]

        turn1 = ScenarioTurn(
            turn_id=0,
            narrative_text="Patient is bleeding.",
            choices=choices,
        )

        turn2 = ScenarioTurn(
            turn_id=1,
            narrative_text="Treatment applied.",
            choices=[
                ChoiceOption(
                    choice_id="monitor",
                    description="Monitor patient",
                    is_correct=True,
                    next_turn_id=2,
                ),
                ChoiceOption(
                    choice_id="evac",
                    description="Evacuate immediately",
                    is_correct=True,
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

        turn3 = ScenarioTurn(
            turn_id=2,
            narrative_text="Patient is stable for transport.",
            choices=[
                ChoiceOption(
                    choice_id="package",
                    description="Package for transport",
                    is_correct=True,
                    next_turn_id=None,
                ),
                ChoiceOption(
                    choice_id="wait",
                    description="Wait for more help",
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
            title="Test Scenario",
            setting="Test Location",
            patient_summary="Test Patient",
            hidden_truth="Test Truth",
            learning_objectives=["Objective 1"],
            turns=[turn1, turn2, turn3],
        )

    def test_scenario_draft_creation(self, sample_scenario):
        """Test creating a complete scenario draft."""
        assert sample_scenario.title == "Test Scenario"
        assert len(sample_scenario.turns) == 3
        assert sample_scenario.get_starting_turn().turn_id == 0

    def test_get_turn(self, sample_scenario):
        """Test retrieving turns by ID."""
        turn = sample_scenario.get_turn(0)
        assert turn is not None
        assert turn.turn_id == 0
        turn = sample_scenario.get_turn(2)
        assert turn is not None
        assert turn.turn_id == 2

    def test_get_turn_not_found(self, sample_scenario):
        """Test retrieving non-existent turn."""
        turn = sample_scenario.get_turn(999)
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
            next_turn_id=2,
        )

        assert choice.choice_id == "apply_tourniquet"
        assert choice.is_correct is True
        assert choice.next_turn_id == 2

    def test_choice_option_end_scenario(self):
        """Test creating a choice that ends the scenario."""
        choice = ChoiceOption(
            choice_id="evacuate",
            description="Evacuate the patient",
            is_correct=True,
            next_turn_id=None,
        )

        assert choice.next_turn_id is None
