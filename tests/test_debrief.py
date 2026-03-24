"""Tests for the debrief agent."""

from unittest.mock import AsyncMock, patch

import pytest

from summit_sim.agents import config as agent_config
from summit_sim.agents.debrief import calculate_score, generate_debrief
from summit_sim.schemas import (
    ChoiceOption,
    DebriefReport,
    ScenarioDraft,
    ScenarioTurn,
)


class TestDebriefAgent:
    """Tests for the debrief agent functionality."""

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

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        turn1 = ScenarioTurn(
            turn_id=1,
            narrative_text="Patient is bleeding.",
            choices=[
                ChoiceOption(
                    choice_id="treat",
                    description="Treat patient",
                    is_correct=True,
                    next_turn_id=2,
                ),
                ChoiceOption(
                    choice_id="wait",
                    description="Wait",
                    is_correct=False,
                    next_turn_id=2,
                ),
            ],
            is_starting_turn=True,
        )

        return ScenarioDraft(
            title="Test Scenario",
            setting="Mountain trail",
            patient_summary="Patient with bleeding",
            hidden_truth="Arterial bleeding",
            learning_objectives=["Control bleeding", "Assess severity"],
            turns=[turn1],
            starting_turn_id=1,
        )

    @pytest.fixture
    def sample_transcript_all_correct(self):
        """Create a transcript with all correct choices."""
        return [
            {
                "turn_id": 1,
                "turn_narrative": "Patient is bleeding.",
                "choice_id": "treat",
                "choice_description": "Treat patient",
                "was_correct": True,
                "feedback": "Good choice!",
                "learning_moments": ["Act quickly"],
                "next_turn_id": 2,
            },
            {
                "turn_id": 2,
                "turn_narrative": "Bleeding controlled.",
                "choice_id": "monitor",
                "choice_description": "Monitor patient",
                "was_correct": True,
                "feedback": "Excellent!",
                "learning_moments": ["Monitor vitals"],
                "next_turn_id": None,
            },
        ]

    @pytest.fixture
    def sample_transcript_mostly_incorrect(self):
        """Create a transcript with mostly incorrect choices."""
        return [
            {
                "turn_id": 1,
                "turn_narrative": "Patient is bleeding.",
                "choice_id": "wait",
                "choice_description": "Wait",
                "was_correct": False,
                "feedback": "Should have acted sooner.",
                "learning_moments": ["Time is critical"],
                "next_turn_id": 2,
            },
            {
                "turn_id": 2,
                "turn_narrative": "Patient worsening.",
                "choice_id": "ignore",
                "choice_description": "Ignore symptoms",
                "was_correct": False,
                "feedback": "Patient needs attention.",
                "learning_moments": ["Assess severity"],
                "next_turn_id": 3,
            },
            {
                "turn_id": 3,
                "turn_narrative": "Emergency situation.",
                "choice_id": "evac",
                "choice_description": "Evacuate",
                "was_correct": True,
                "feedback": "Correct decision.",
                "learning_moments": ["Know when to evac"],
                "next_turn_id": None,
            },
        ]

    @pytest.mark.asyncio
    async def test_generate_debrief_returns_report(
        self, sample_scenario, sample_transcript_all_correct
    ):
        """Test agent returns DebriefReport."""
        mock_report = DebriefReport(
            summary="Excellent performance",
            key_mistakes=[],
            strong_actions=["Quick response", "Proper assessment"],
            best_next_actions=["Practice more scenarios"],
            teaching_points=["Time management"],
            completion_status="pass",
            final_score=100.0,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = AsyncMock(output=mock_report)
            mock_agent_class.return_value = mock_agent

            result = await generate_debrief(
                sample_transcript_all_correct, sample_scenario, "scn-test123"
            )

        assert isinstance(result, DebriefReport)
        assert result.summary == "Excellent performance"
        assert result.completion_status == "pass"

    @pytest.mark.asyncio
    async def test_debrief_fail_status(
        self, sample_scenario, sample_transcript_mostly_incorrect
    ):
        """Test report shows fail when mistakes present."""
        mock_report = DebriefReport(
            summary="Needs improvement",
            key_mistakes=["Delayed treatment", "Ignored symptoms"],
            strong_actions=["Correct evacuation decision"],
            best_next_actions=["Practice triage", "Review protocols"],
            teaching_points=["Time sensitivity", "Assessment skills"],
            completion_status="fail",
            final_score=33.33,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = AsyncMock(output=mock_report)
            mock_agent_class.return_value = mock_agent

            result = await generate_debrief(
                sample_transcript_mostly_incorrect, sample_scenario, "scn-test456"
            )

        assert result.completion_status == "fail"
        assert result.final_score < 70
        assert len(result.key_mistakes) > 0

    @pytest.mark.asyncio
    async def test_final_score_calculation(
        self, sample_scenario, sample_transcript_mostly_incorrect
    ):
        """Test score is percentage of correct choices."""
        mock_report = DebriefReport(
            summary="Test score",
            key_mistakes=["Mistake 1", "Mistake 2"],
            strong_actions=["Good action"],
            best_next_actions=["Do better"],
            teaching_points=["Learn this"],
            completion_status="fail",
            final_score=33.33,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = AsyncMock(output=mock_report)
            mock_agent_class.return_value = mock_agent

            result = await generate_debrief(
                sample_transcript_mostly_incorrect, sample_scenario, "scn-test789"
            )

        assert result.final_score == 33.33

    @pytest.mark.asyncio
    async def test_debrief_includes_mistakes(
        self, sample_scenario, sample_transcript_mostly_incorrect
    ):
        """Test report identifies specific mistakes."""
        mock_report = DebriefReport(
            summary="Analysis complete",
            key_mistakes=["Delayed treatment", "Ignored symptoms"],
            strong_actions=["Final evacuation"],
            best_next_actions=["Practice assessment"],
            teaching_points=["Time management"],
            completion_status="fail",
            final_score=33.33,
        )

        with patch("summit_sim.agents.config.Agent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = AsyncMock(output=mock_report)
            mock_agent_class.return_value = mock_agent

            result = await generate_debrief(
                sample_transcript_mostly_incorrect, sample_scenario, "scn-testabc"
            )

        assert "Delayed treatment" in result.key_mistakes
        assert "Ignored symptoms" in result.key_mistakes


class TestCalculateScore:
    """Tests for the calculate_score function."""

    def test_calculate_score_all_correct(self):
        """Test score with all correct choices."""
        transcript = [
            {"was_correct": True},
            {"was_correct": True},
            {"was_correct": True},
        ]
        score = calculate_score(transcript)
        assert score == 100.0

    def test_calculate_score_all_incorrect(self):
        """Test score with all incorrect choices."""
        transcript = [
            {"was_correct": False},
            {"was_correct": False},
        ]
        score = calculate_score(transcript)
        assert score == 0.0

    def test_calculate_score_mixed(self):
        """Test score with mixed correctness."""
        transcript = [
            {"was_correct": True},
            {"was_correct": False},
            {"was_correct": True},
        ]
        score = calculate_score(transcript)
        assert score == pytest.approx(66.67, rel=0.01)

    def test_calculate_score_empty_transcript(self):
        """Test score with empty transcript."""
        transcript = []
        score = calculate_score(transcript)
        assert score == 0.0

    def test_calculate_score_single_correct(self):
        """Test score with single correct choice."""
        transcript = [{"was_correct": True}]
        score = calculate_score(transcript)
        assert score == 100.0

    def test_calculate_score_single_incorrect(self):
        """Test score with single incorrect choice."""
        transcript = [{"was_correct": False}]
        score = calculate_score(transcript)
        assert score == 0.0
