"""Tests for the action responder agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summit_sim.agents.action_responder import (
    AGENT_NAME,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _format_conversation_history,
    process_action,
)
from summit_sim.graphs.simulation import SimulationState
from summit_sim.schemas import DynamicTurnResult, ScenarioDraft, TranscriptEntry


class TestFormatConversationHistory:
    """Tests for _format_conversation_history function."""

    def test_empty_history(self):
        """Test formatting empty transcript."""
        result = _format_conversation_history([])
        assert result == "Initial turn - no previous actions."

    def test_single_entry(self):
        """Test formatting single transcript entry."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="You check the patient's airway...",
                student_action="I check the patient's airway",
                was_correct=True,
                feedback="Good first step in assessing patient",
                learning_moments=["Airway assessment"],
            )
        ]

        result = _format_conversation_history(transcript)

        assert "Student: I check the patient's airway" in result
        assert "AI: You check the patient's airway..." in result

    def test_multiple_entries(self):
        """Test formatting multiple transcript entries."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Narrative 1",
                student_action="Action 1",
                was_correct=True,
                feedback="Feedback 1",
                learning_moments=["Point 1"],
            ),
            TranscriptEntry(
                turn_id=2,
                turn_narrative="Narrative 2",
                student_action="Action 2",
                was_correct=False,
                feedback="Feedback 2",
                learning_moments=["Point 2"],
            ),
        ]

        result = _format_conversation_history(transcript)

        assert "Student: Action 1" in result
        assert "Student: Action 2" in result
        assert "AI: Narrative 1" in result
        assert "AI: Narrative 2" in result

    def test_history_truncation(self):
        """Test that only last 5 entries are included."""
        transcript = [
            TranscriptEntry(
                turn_id=i,
                turn_narrative=f"Narrative {i}",
                student_action=f"Action {i}",
                was_correct=True,
                feedback=f"Feedback {i}",
                learning_moments=[f"Point {i}"],
            )
            for i in range(1, 11)
        ]

        result = _format_conversation_history(transcript)

        # Should only show last 5 entries (turns 6-10)
        assert "Student: Action 6" in result
        assert "Student: Action 10" in result
        # Check that early entries (1-5) are not present as student actions
        assert "Student: Action 1\n" not in result
        assert "Student: Action 5\n" not in result

    def test_feedback_truncation(self):
        """Test that long feedback is truncated."""
        transcript = [
            TranscriptEntry(
                turn_id=1,
                turn_narrative="Narrative",
                student_action="Action",
                was_correct=True,
                feedback="A" * 200,  # Long feedback
                learning_moments=["Point"],
            )
        ]

        result = _format_conversation_history(transcript)

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
    def sample_simulation_state(self):
        """Create a sample simulation state for testing."""
        scenario = ScenarioDraft(
            title="Test Emergency",
            setting="Mountain trail at 8000ft",
            patient_summary="30yo male with head injury",
            hidden_truth="Severe concussion with potential skull fracture",
            learning_objectives=["Assess consciousness", "Stabilize head"],
            initial_narrative="You find an unconscious hiker...",
            hidden_state="Patient unconscious, GCS 8",
            scene_state="Clear weather, 2 hours to sunset",
        )

        return SimulationState(
            scenario=scenario,
            transcript=[],
            turn_count=0,
            is_complete=False,
            action_result=None,
            scenario_id="test-scenario-123",
            debrief_report=None,
            hidden_state=scenario.hidden_state,
        )

    @pytest.mark.asyncio
    async def test_process_action_success(
        self, sample_scenario, sample_simulation_state
    ):
        """Test successful action processing."""
        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.3,
            feedback="Good approach to stabilize the patient",
            narrative_text="You carefully stabilize the patient's head...",
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
                simulation_state=sample_simulation_state,
                max_turns=5,
            )

            assert result == expected_result
            assert result.was_correct is True

    @pytest.mark.asyncio
    async def test_process_action_with_transcript_history(self, sample_scenario):
        """Test action processing with transcript history."""
        transcript_entry = TranscriptEntry(
            turn_id=1,
            turn_narrative="Airway is clear",
            student_action="Checked airway",
            was_correct=True,
            feedback="Good first step",
            learning_moments=["Airway assessment"],
        )

        state_with_history = SimulationState(
            scenario=sample_scenario,
            transcript=[transcript_entry],
            turn_count=1,
            is_complete=False,
            action_result=None,
            scenario_id="test-scenario-123",
            debrief_report=None,
            hidden_state=sample_scenario.hidden_state,
        )

        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=0.5,
            feedback="Continuing good assessment",
            narrative_text="Patient responding well...",
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
                simulation_state=state_with_history,
                max_turns=5,
            )

            assert result == expected_result

    @pytest.mark.asyncio
    async def test_process_action_completes_scenario(self, sample_scenario):
        """Test action that completes the scenario."""
        state = SimulationState(
            scenario=sample_scenario,
            transcript=[],
            turn_count=3,
            is_complete=False,
            action_result={"completion_score": 0.6},
            scenario_id="test-scenario-123",
            debrief_report=None,
            hidden_state="Patient stable, evacuation ready",
        )

        expected_result = DynamicTurnResult(
            was_correct=True,
            completion_score=1.0,
            feedback="Excellent work! Patient successfully evacuated.",
            narrative_text="The helicopter lifts off with the patient...",
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
                simulation_state=state,
                max_turns=5,
            )

            assert result.completion_score == 1.0

    @pytest.mark.asyncio
    async def test_process_action_incorrect_action(
        self, sample_scenario, sample_simulation_state
    ):
        """Test processing an incorrect student action."""
        expected_result = DynamicTurnResult(
            was_correct=False,
            completion_score=0.1,
            feedback="Moving the patient with a potential spinal injury is dangerous",
            narrative_text="As you move the patient, they cry out in pain...",
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
                simulation_state=sample_simulation_state,
                max_turns=5,
            )

            assert result.was_correct is False

    def test_agent_name_constant(self):
        """Test that AGENT_NAME constant is set correctly."""
        assert AGENT_NAME == "action-responder"

    def test_system_prompt_defined(self):
        """Test that system prompt is defined."""
        assert len(SYSTEM_PROMPT) > 0
        assert "wilderness first responder" in SYSTEM_PROMPT.lower()
        assert "was_correct" in SYSTEM_PROMPT

    def test_user_prompt_template_defined(self):
        """Test that user prompt template is defined."""
        assert len(USER_PROMPT_TEMPLATE) > 0
        assert "{{title}}" in USER_PROMPT_TEMPLATE
        assert "{{student_action}}" in USER_PROMPT_TEMPLATE
        assert "{{hidden_state}}" in USER_PROMPT_TEMPLATE
