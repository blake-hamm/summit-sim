"""Tests for the debrief agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mlflow.entities.model_registry.prompt_version import PromptVersion

from summit_sim.agents.debrief import (
    AGENT_NAME,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _build_debrief_prompt,
    _format_scenario_context,
    _format_transcript_summary,
    calculate_score,
    generate_debrief,
)
from summit_sim.graphs.utils import TranscriptEntry
from summit_sim.schemas import DebriefReport, ScenarioDraft


class TestAgentConstants:
    """Tests for agent constants."""

    def test_agent_name(self):
        """Test that AGENT_NAME is set correctly."""
        assert AGENT_NAME == "debrief"

    def test_system_prompt_contains_key_elements(self):
        """Test that system prompt contains required elements."""
        assert "wilderness first aid" in SYSTEM_PROMPT.lower()
        assert "debrief" in SYSTEM_PROMPT.lower()
        assert "scoring" in SYSTEM_PROMPT.lower()

    def test_user_prompt_template_contains_placeholders(self):
        """Test that user prompt template has all required placeholders."""
        assert "{{scenario_context}}" in USER_PROMPT_TEMPLATE
        assert "{{scenario_id}}" in USER_PROMPT_TEMPLATE
        assert "{{total_turns}}" in USER_PROMPT_TEMPLATE
        assert "{{transcript_summary}}" in USER_PROMPT_TEMPLATE
        assert "{{correct_count}}" in USER_PROMPT_TEMPLATE
        assert "{{incorrect_count}}" in USER_PROMPT_TEMPLATE
        assert "{{score}}" in USER_PROMPT_TEMPLATE


class TestCalculateScore:
    """Tests for calculate_score function."""

    def test_empty_transcript(self):
        """Test score calculation with empty transcript."""
        score = calculate_score([])
        assert score == 0.0

    def test_all_correct(self):
        """Test score with all correct actions."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Good",
                learning_moments=[],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Turn 2",
                student_action="Action 2",
                was_correct=True,
                feedback="Good",
                learning_moments=[],
            ),
        ]
        score = calculate_score(transcript)
        assert score == 100.0

    def test_all_incorrect(self):
        """Test score with all incorrect actions."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=False,
                feedback="Bad",
                learning_moments=[],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Turn 2",
                student_action="Action 2",
                was_correct=False,
                feedback="Bad",
                learning_moments=[],
            ),
        ]
        score = calculate_score(transcript)
        assert score == 0.0

    def test_mixed_results(self):
        """Test score with mixed correct/incorrect actions."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Good",
                learning_moments=[],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Turn 2",
                student_action="Action 2",
                was_correct=False,
                feedback="Bad",
                learning_moments=[],
            ),
            TranscriptEntry(
                turn_id=3,
                turn_narrative="Turn 3",
                student_action="Action 3",
                was_correct=True,
                feedback="Good",
                learning_moments=[],
            ),
        ]
        score = calculate_score(transcript)
        assert score == pytest.approx(66.67, rel=0.01)

    def test_single_correct(self):
        """Test score with single correct action."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Good",
                learning_moments=[],
            ),
        ]
        score = calculate_score(transcript)
        assert score == 100.0

    def test_single_incorrect(self):
        """Test score with single incorrect action."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=False,
                feedback="Bad",
                learning_moments=[],
            ),
        ]
        score = calculate_score(transcript)
        assert score == 0.0


class TestFormatScenarioContext:
    """Tests for _format_scenario_context function."""

    def test_format_basic_scenario(self):
        """Test formatting basic scenario information."""
        scenario = ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male with injury",
            hidden_truth="Fractured leg",
            learning_objectives=["Assess injury", "Immobilize limb"],
            initial_narrative="You find a hiker...",
            hidden_state="Hidden",
            scene_state="Scene",
        )

        result = _format_scenario_context(scenario)

        assert "Title: Test Emergency" in result
        assert "Setting: Mountain trail" in result
        assert "Patient: 30yo male with injury" in result
        assert "Learning Objectives: Assess injury, Immobilize limb" in result


class TestFormatTranscriptSummary:
    """Tests for _format_transcript_summary function."""

    def test_format_empty_transcript(self):
        """Test formatting empty transcript."""
        result = _format_transcript_summary([])
        assert result == ""

    def test_format_single_entry(self):
        """Test formatting single transcript entry."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1 narrative",
                student_action="I check vitals",
                was_correct=True,
                feedback="Good first step",
                learning_moments=["Lesson 1"],
            ),
        ]

        result = _format_transcript_summary(transcript)

        assert "Turn 1: CORRECT" in result
        assert "Action: I check vitals" in result
        assert "Feedback: Good first step" in result

    def test_format_multiple_entries(self):
        """Test formatting multiple transcript entries."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Good",
                learning_moments=[],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Turn 2",
                student_action="Action 2",
                was_correct=False,
                feedback="Should have acted differently",
                learning_moments=["Lesson"],
            ),
        ]

        result = _format_transcript_summary(transcript)

        assert "Turn 1: CORRECT" in result
        assert "Turn 2: INCORRECT" in result
        assert "Action: Action 1" in result
        assert "Action: Action 2" in result


class TestBuildDebriefPrompt:
    """Tests for _build_debrief_prompt function."""

    def test_build_prompt(self):
        """Test building debrief prompt."""
        scenario = ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Fracture",
            learning_objectives=["Learn"],
            initial_narrative="Start",
            hidden_state="Hidden",
            scene_state="Scene",
        )

        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Good",
                learning_moments=["Lesson"],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Turn 2",
                student_action="Action 2",
                was_correct=False,
                feedback="Bad",
                learning_moments=["Lesson 2"],
            ),
        ]

        mock_user_prompt = MagicMock(spec=PromptVersion)
        mock_user_prompt.format.return_value = "formatted prompt"

        result = _build_debrief_prompt(
            transcript, scenario, "scn-123", mock_user_prompt
        )

        assert result == "formatted prompt"
        # Verify format was called with correct kwargs
        call_kwargs = mock_user_prompt.format.call_args.kwargs
        assert call_kwargs["scenario_id"] == "scn-123"
        assert call_kwargs["total_turns"] == 2
        assert call_kwargs["correct_count"] == 1
        assert call_kwargs["incorrect_count"] == 1
        assert "50.0" in call_kwargs["score"]


class TestGenerateDebrief:
    """Tests for generate_debrief function."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario."""
        return ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail",
            patient_summary="30yo male",
            hidden_truth="Fracture",
            learning_objectives=["Learn"],
            initial_narrative="Start",
            hidden_state="Hidden",
            scene_state="Scene",
        )

    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript."""
        return [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Good action",
                was_correct=True,
                feedback="Well done",
                learning_moments=["Lesson 1"],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Turn 2",
                student_action="Bad action",
                was_correct=False,
                feedback="Should improve",
                learning_moments=["Lesson 2"],
            ),
        ]

    @pytest.mark.asyncio
    async def test_generate_debrief_success(self, sample_scenario, sample_transcript):
        """Test successful debrief generation."""
        expected_report = DebriefReport(
            summary="Good performance overall",
            key_mistakes=["One mistake"],
            strong_actions=["Good initial assessment"],
            best_next_actions=["Practice more scenarios"],
            teaching_points=["Key concept"],
            completion_status="pass",
            final_score=50.0,
        )

        mock_response = MagicMock()
        mock_response.output = expected_report

        with patch("summit_sim.agents.debrief.setup_agent_and_prompts") as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            class MockPrompt:
                def format(self, **_kwargs) -> str:
                    return "formatted prompt"

            mock_user_prompt = MockPrompt()
            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await generate_debrief(
                transcript=sample_transcript,
                scenario_draft=sample_scenario,
                scenario_id="scn-test-123",
            )

            assert result == expected_report
            assert result.completion_status == "pass"
            assert result.final_score == 50.0

    @pytest.mark.asyncio
    async def test_generate_debrief_fail_status(self, sample_scenario):
        """Test debrief with fail status."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Turn 1",
                student_action="Bad action",
                was_correct=False,
                feedback="Critical error",
                learning_moments=["Important lesson"],
            ),
        ]

        expected_report = DebriefReport(
            summary="Needs improvement",
            key_mistakes=["Critical mistake"],
            strong_actions=[],
            best_next_actions=["Review protocols", "Practice more"],
            teaching_points=["Review basics"],
            completion_status="fail",
            final_score=0.0,
        )

        mock_response = MagicMock()
        mock_response.output = expected_report

        with patch("summit_sim.agents.debrief.setup_agent_and_prompts") as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            class MockPrompt:
                def format(self, **_kwargs) -> str:
                    return "formatted prompt"

            mock_user_prompt = MockPrompt()
            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await generate_debrief(
                transcript=transcript,
                scenario_draft=sample_scenario,
                scenario_id="scn-fail-456",
            )

            assert result.completion_status == "fail"
            assert result.final_score < 70

    @pytest.mark.asyncio
    async def test_generate_debrief_perfect_score(self, sample_scenario):
        """Test debrief with perfect score."""
        transcript = [
            TranscriptEntry(
                turn_id=i,
                turn_narrative=f"Turn {i}",
                student_action=f"Perfect action {i}",
                was_correct=True,
                feedback="Excellent!",
                learning_moments=[f"Lesson {i}"],
            )
            for i in range(1, 4)
        ]

        expected_report = DebriefReport(
            summary="Excellent performance!",
            key_mistakes=[],
            strong_actions=["All actions were correct"],
            best_next_actions=["Consider advanced training"],
            teaching_points=["Great job!"],
            completion_status="pass",
            final_score=100.0,
        )

        mock_response = MagicMock()
        mock_response.output = expected_report

        with patch("summit_sim.agents.debrief.setup_agent_and_prompts") as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            class MockPrompt:
                def format(self, **_kwargs) -> str:
                    return "formatted prompt"

            mock_user_prompt = MockPrompt()
            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await generate_debrief(
                transcript=transcript,
                scenario_draft=sample_scenario,
                scenario_id="scn-perfect-789",
            )

            assert result.completion_status == "pass"
            assert result.final_score == 100.0
