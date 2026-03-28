"""Tests for the action responder agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summit_sim.agents.action_responder import (
    AGENT_NAME,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    TurnContext,
    _format_transcript_history,
    process_action,
)
from summit_sim.schemas import DynamicTurnResult, ScenarioDraft


class TestTurnContext:
    """Tests for TurnContext dataclass."""

    def test_turn_context_creation(self):
        """Test creating a TurnContext instance."""
        context = TurnContext(
            hidden_state="Patient vitals stable",
            scene_state="Clear weather",
            transcript_history=[{"action": "checked pulse"}],
            turn_count=1,
            max_turns=5,
        )

        assert context.hidden_state == "Patient vitals stable"
        assert context.scene_state == "Clear weather"
        assert len(context.transcript_history) == 1
        assert context.turn_count == 1
        assert context.max_turns == 5

    def test_turn_context_empty_history(self):
        """Test TurnContext with empty transcript history."""
        context = TurnContext(
            hidden_state="Initial state",
            scene_state="Initial scene",
            transcript_history=[],
            turn_count=0,
            max_turns=5,
        )

        assert context.transcript_history == []
        assert context.turn_count == 0


class TestFormatTranscriptHistory:
    """Tests for _format_transcript_history function."""

    def test_empty_history(self):
        """Test formatting empty transcript history."""
        result = _format_transcript_history([])
        assert result == "No previous actions (initial turn)."

    def test_single_entry(self):
        """Test formatting single transcript entry."""
        history = [
            {
                "action": "I check the patient's airway",
                "feedback": "Good first step in assessing patient",
                "narrative": "You kneel beside the patient and check their airway...",
                "was_correct": True,
            }
        ]

        result = _format_transcript_history(history)

        assert "Turn 1:" in result
        assert "Student: I check the patient's airway" in result
        assert "Good first step" in result

    def test_multiple_entries(self):
        """Test formatting multiple transcript entries."""
        history = [
            {
                "action": "Action 1",
                "feedback": "Feedback 1",
                "narrative": "Narrative 1",
                "was_correct": True,
            },
            {
                "action": "Action 2",
                "feedback": "Feedback 2",
                "narrative": "Narrative 2",
                "was_correct": False,
            },
        ]

        result = _format_transcript_history(history)

        assert "Turn 1:" in result
        assert "Turn 2:" in result
        assert "Action 1" in result
        assert "Action 2" in result

    def test_history_truncation(self):
        """Test that only last 5 entries are included."""
        history = [
            {
                "action": f"Action {i}",
                "feedback": f"Feedback {i}",
                "narrative": f"Narrative {i}",
                "was_correct": True,
            }
            for i in range(10)
        ]

        result = _format_transcript_history(history)

        # Should only show 5 entries (renumbered 1-5)
        assert "Turn 1:" in result
        assert "Turn 5:" in result
        assert "Turn 6:" not in result  # Only 5 entries shown
        # Check that last actions are shown
        assert "Action 5" in result
        assert "Action 9" in result

    def test_long_text_truncation(self):
        """Test that long text is truncated."""
        history = [
            {
                "action": "Short action",
                "feedback": "A" * 200,  # Long feedback
                "narrative": "B" * 300,  # Long narrative
                "was_correct": True,
            }
        ]

        result = _format_transcript_history(history)

        # Check that feedback is truncated (showing ...)
        assert "..." in result


class TestProcessAction:
    """Tests for process_action function."""

    @pytest.fixture
    def sample_scenario(self):
        """Create a sample scenario for testing."""
        return ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail at 8000ft",
            patient_summary="30yo male with head injury",
            hidden_truth="Severe concussion with potential skull fracture",
            learning_objectives=["Assess consciousness", "Stabilize head"],
            initial_narrative="You find an unconscious hiker...",
            hidden_state="Patient unconscious, GCS 8",
            scene_state="Clear weather, 2 hours to sunset",
        )

    @pytest.fixture
    def sample_context(self):
        """Create a sample turn context for testing."""
        return TurnContext(
            hidden_state="Patient unconscious, GCS 8",
            scene_state="Clear weather, 2 hours to sunset",
            transcript_history=[],
            turn_count=1,
            max_turns=5,
        )

    @pytest.mark.asyncio
    async def test_process_action_success(self, sample_scenario, sample_context):
        """Test successful action processing."""
        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.3,
            is_complete=False,
            feedback="Good approach to stabilize the patient",
            narrative_text="You carefully stabilize the patient's head...",
            updated_hidden_state="Patient vitals stable, head stabilized",
            updated_scene_state="Clear weather, 2 hours to sunset, patient stable",
        )

        mock_response = MagicMock()
        mock_response.output = expected_result

        with patch(
            "summit_sim.agents.action_responder.setup_agent_and_prompts"
        ) as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            mock_user_prompt = MagicMock()
            mock_user_prompt.format.return_value = "formatted prompt"

            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await process_action(
                student_action="I stabilize the patient's head",
                scenario=sample_scenario,
                context=sample_context,
            )

            assert result == expected_result
            assert result.was_correct is True
            assert result.is_complete is False

    @pytest.mark.asyncio
    async def test_process_action_with_transcript_history(self, sample_scenario):
        """Test action processing with transcript history."""
        context_with_history = TurnContext(
            hidden_state="Patient breathing",
            scene_state="Weather clear",
            transcript_history=[
                {
                    "action": "Checked airway",
                    "feedback": "Good first step",
                    "narrative": "Airway is clear",
                    "was_correct": True,
                }
            ],
            turn_count=2,
            max_turns=5,
        )

        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.5,
            is_complete=False,
            feedback="Continuing good assessment",
            narrative_text="Patient responding well...",
            updated_hidden_state="Stable vitals",
            updated_scene_state="Weather unchanged",
        )

        mock_response = MagicMock()
        mock_response.output = expected_result

        with patch(
            "summit_sim.agents.action_responder.setup_agent_and_prompts"
        ) as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            mock_user_prompt = MagicMock()
            mock_user_prompt.format.return_value = "formatted prompt with history"

            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await process_action(
                student_action="I check breathing",
                scenario=sample_scenario,
                context=context_with_history,
            )

            assert result == expected_result

    @pytest.mark.asyncio
    async def test_process_action_completes_scenario(self, sample_scenario):
        """Test action that completes the scenario."""
        context = TurnContext(
            hidden_state="Patient stable, evacuation ready",
            scene_state="Helicopter arrived",
            transcript_history=[],
            turn_count=4,
            max_turns=5,
        )

        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=1.0,
            is_complete=True,
            feedback="Excellent work! Patient successfully evacuated.",
            narrative_text="The helicopter lifts off with the patient...",
            updated_hidden_state="Patient evacuated to medical facility",
            updated_scene_state="Rescue complete, returning to base",
        )

        mock_response = MagicMock()
        mock_response.output = expected_result

        with patch(
            "summit_sim.agents.action_responder.setup_agent_and_prompts"
        ) as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            mock_user_prompt = MagicMock()
            mock_user_prompt.format.return_value = "formatted prompt"

            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await process_action(
                student_action="I load patient into helicopter",
                scenario=sample_scenario,
                context=context,
            )

            assert result.is_complete is True
            assert result.completion_score == 1.0

    @pytest.mark.asyncio
    async def test_process_action_incorrect_action(
        self, sample_scenario, sample_context
    ):
        """Test processing an incorrect student action."""
        expected_result = DynamicTurnResult(
            was_correct=False,
            completion_score=0.1,
            is_complete=False,
            feedback="Moving the patient with a potential spinal injury is dangerous",
            narrative_text="As you move the patient, they cry out in pain...",
            updated_hidden_state="Patient condition worsened, increased pain",
            updated_scene_state="Clear weather, patient distressed",
        )

        mock_response = MagicMock()
        mock_response.output = expected_result

        with patch(
            "summit_sim.agents.action_responder.setup_agent_and_prompts"
        ) as mock_setup:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_response

            mock_user_prompt = MagicMock()
            mock_user_prompt.format.return_value = "formatted prompt"

            mock_setup.return_value = (mock_agent, mock_user_prompt)

            result = await process_action(
                student_action="I move the patient quickly",
                scenario=sample_scenario,
                context=sample_context,
            )

            assert result.was_correct is False
            assert (
                "dangerous" in result.feedback.lower()
                or "pain" in result.narrative_text.lower()
            )

    def test_agent_name_constant(self):
        """Test that AGENT_NAME constant is set correctly."""
        assert AGENT_NAME == "action-responder"

    def test_system_prompt_defined(self):
        """Test that system prompt is defined."""
        assert len(SYSTEM_PROMPT) > 0
        assert "wilderness first aid" in SYSTEM_PROMPT.lower()
        assert "was_correct" in SYSTEM_PROMPT
        assert "narrative_text" in SYSTEM_PROMPT

    def test_user_prompt_template_defined(self):
        """Test that user prompt template is defined."""
        assert len(USER_PROMPT_TEMPLATE) > 0
        assert "{{title}}" in USER_PROMPT_TEMPLATE
        assert "{{student_action}}" in USER_PROMPT_TEMPLATE
        assert "{{hidden_state}}" in USER_PROMPT_TEMPLATE
