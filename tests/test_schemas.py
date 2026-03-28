"""Tests for Pydantic schemas."""

import pytest

from summit_sim.schemas import (
    DynamicTurnResult,
    ScenarioConfig,
    ScenarioDraft,
)


class TestScenarioConfig:
    """Tests for ScenarioConfig schema."""

    def test_scenario_config_creation(self):
        """Test creating minimal scenario configuration."""
        config = ScenarioConfig(
            primary_focus="Trauma",
            environment="Alpine/Mountain",
            available_personnel="Small Group (3-5)",
            evac_distance="Remote (1 day)",
            complexity="Standard",
        )
        assert config.primary_focus == "Trauma"
        assert config.environment == "Alpine/Mountain"
        assert config.complexity == "Standard"

    def test_scenario_config_all_focus_areas(self):
        """Test valid primary focus values."""
        for focus in ["Trauma", "Medical", "Environmental", "Mixed"]:
            config = ScenarioConfig(
                primary_focus=focus,
                environment="Forest/Trail",
                available_personnel="Partner (2)",
                evac_distance="Short (< 2 hours)",
                complexity="Standard",
            )
            assert config.primary_focus == focus

    def test_scenario_config_all_complexity_levels(self):
        """Test valid complexity values."""
        for complexity in ["Standard", "Complicated", "Critical"]:
            config = ScenarioConfig(
                primary_focus="Medical",
                environment="Winter/Snow",
                available_personnel="Large Expedition (6+)",
                evac_distance="Expedition (2+ days)",
                complexity=complexity,
            )
            assert config.complexity == complexity


class TestScenarioDraft:
    """Tests for ScenarioDraft schema."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        return ScenarioDraft(
            title="Test Scenario",
            setting="Test Location",
            patient_summary="Test Patient",
            hidden_truth="Test Truth",
            learning_objectives=["Objective 1", "Objective 2"],
            initial_narrative=(
                "You arrive at the scene and see a patient lying on the ground."
            ),
            hidden_state="Patient has weak pulse and labored breathing",
            scene_state="Clear weather, 65F temperature",
        )

    def test_scenario_draft_creation(self, sample_scenario):
        """Test creating a complete scenario draft."""
        assert sample_scenario.title == "Test Scenario"
        assert (
            sample_scenario.initial_narrative
            == "You arrive at the scene and see a patient lying on the ground."
        )
        assert "weak pulse" in sample_scenario.hidden_state
        assert "clear" in sample_scenario.scene_state.lower()

    def test_scenario_draft_minimal(self):
        """Test creating scenario with minimal fields."""
        scenario = ScenarioDraft(
            title="Minimal Scenario",
            setting="Somewhere",
            patient_summary="A patient",
            hidden_truth="Something",
            learning_objectives=["Learn something", "Apply knowledge"],
            initial_narrative="The scene opens.",
            hidden_state="Initial hidden state",
            scene_state="Initial scene state",
        )
        assert scenario.hidden_state == "Initial hidden state"
        assert scenario.scene_state == "Initial scene state"


class TestDynamicTurnResult:
    """Tests for DynamicTurnResult schema."""

    def test_dynamic_turn_result_creation(self):
        """Test creating a dynamic turn result."""
        result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.75,
            is_complete=False,
            feedback="Good job assessing the patient.",
            narrative_text="You check the patient's pulse and find it weak.",
            updated_hidden_state="Patient has weak pulse and BP 90/60",
            updated_scene_state="5 minutes elapsed since arrival",
        )

        assert result.was_correct is True
        assert result.completion_score == 0.75
        assert result.is_complete is False
        assert result.feedback == "Good job assessing the patient."
        assert (
            result.narrative_text == "You check the patient's pulse and find it weak."
        )
        assert "weak pulse" in result.updated_hidden_state

    def test_dynamic_turn_result_complete(self):
        """Test creating a completed scenario result."""
        result = DynamicTurnResult(
            was_correct=True,
            completion_score=1.0,
            is_complete=True,
            feedback="Scenario complete. Patient evacuated successfully.",
            narrative_text="The evacuation helicopter arrives and takes the patient.",
            updated_hidden_state="Patient evacuated to medical facility",
            updated_scene_state="Rescue operation complete",
        )

        assert result.is_complete is True
        assert result.completion_score == 1.0

    def test_dynamic_turn_result_score_bounds(self):
        """Test that completion_score must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            DynamicTurnResult(
                was_correct=True,
                completion_score=1.5,
                is_complete=False,
                feedback="Test",
                narrative_text="Test",
                updated_hidden_state="Test state",
                updated_scene_state="Test scene",
            )

        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 0"
        ):
            DynamicTurnResult(
                was_correct=True,
                completion_score=-0.1,
                is_complete=False,
                feedback="Test",
                narrative_text="Test",
                updated_hidden_state="Test state",
                updated_scene_state="Test scene",
            )
