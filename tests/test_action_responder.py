"""Tests for the action responder agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summit_sim.agents.action_responder import (
    AGENT_NAME,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    action_response_agent,
)
from summit_sim.schemas import ActionResponseInput, DynamicTurnResult


class TestActionResponseAgent:
    """Tests for action_response_agent function."""

    @pytest.fixture
    def sample_input(self):
        """Create a sample ActionResponseInput for testing."""
        return ActionResponseInput(
            student_action="I stabilize the patient's head",
            scenario_title="Test Emergency",
            scenario_setting="Mountain trail at 8000ft",
            patient_summary="30yo male with head injury",
            hidden_truth="Severe concussion with potential skull fracture",
            learning_objectives=["Assess consciousness", "Stabilize head"],
            transcript=[],
            previous_score=0.0,
            turn_count=1,
            max_turns=5,
            hidden_state="Patient unconscious, GCS 8",
        )

    @pytest.fixture
    def sample_input_with_history(self):
        """Create a sample ActionResponseInput with conversation history."""
        return ActionResponseInput(
            student_action="I check breathing",
            scenario_title="Test Emergency",
            scenario_setting="Mountain trail at 8000ft",
            patient_summary="30yo male with head injury",
            hidden_truth="Severe concussion with potential skull fracture",
            learning_objectives=["Assess consciousness", "Stabilize head"],
            transcript=[
                {
                    "turn_id": 1,
                    "student_action": "Checked airway",
                    "turn_narrative": "Airway is clear",
                    "was_correct": True,
                    "feedback": "Good first step",
                }
            ],
            previous_score=0.2,
            turn_count=2,
            max_turns=5,
            hidden_state="Patient unconscious, GCS 8",
        )

    @pytest.mark.asyncio
    async def test_action_response_agent_success(self, sample_input):
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

            result = await action_response_agent(sample_input)

            assert result == expected_result
            assert result.was_correct is True

    @pytest.mark.asyncio
    async def test_action_response_agent_with_history(self, sample_input_with_history):
        """Test action processing with conversation history."""
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

            result = await action_response_agent(sample_input_with_history)

            assert result == expected_result

    @pytest.mark.asyncio
    async def test_action_response_agent_completes_scenario(self):
        """Test action that completes the scenario."""
        input_data = ActionResponseInput(
            student_action="I load patient into helicopter",
            scenario_title="Test Emergency",
            scenario_setting="Mountain trail at 8000ft",
            patient_summary="30yo male with head injury",
            hidden_truth="Severe concussion with potential skull fracture",
            learning_objectives=["Assess consciousness", "Stabilize head"],
            transcript=[
                {
                    "turn_id": 1,
                    "student_action": "Stabilized patient",
                    "turn_narrative": "Patient is stable",
                    "was_correct": True,
                    "feedback": "Good work",
                },
                {
                    "turn_id": 2,
                    "student_action": "Called for evacuation",
                    "turn_narrative": "Helicopter is on the way",
                    "was_correct": True,
                    "feedback": "Correct decision",
                },
            ],
            previous_score=0.6,
            turn_count=4,
            max_turns=5,
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

            result = await action_response_agent(input_data)

            assert result.completion_score == 1.0

    @pytest.mark.asyncio
    async def test_action_response_agent_incorrect_action(self):
        """Test processing an incorrect student action."""
        input_data = ActionResponseInput(
            student_action="I move the patient quickly",
            scenario_title="Test Emergency",
            scenario_setting="Mountain trail at 8000ft",
            patient_summary="30yo male with head injury",
            hidden_truth="Severe concussion with potential skull fracture",
            learning_objectives=["Assess consciousness", "Stabilize head"],
            transcript=[],
            previous_score=0.0,
            turn_count=1,
            max_turns=5,
            hidden_state="Patient unconscious, GCS 8",
        )

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

            result = await action_response_agent(input_data)

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


class TestActionResponseInput:
    """Tests for ActionResponseInput model."""

    def test_create_input(self):
        """Test creating ActionResponseInput."""
        input_data = ActionResponseInput(
            student_action="I check vitals",
            scenario_title="Mountain Emergency",
            scenario_setting="Alpine ridge at 12000ft",
            patient_summary="45yo female with chest pain",
            hidden_truth="High altitude pulmonary edema",
            learning_objectives=["Assess breathing", "Recognize HAPE"],
            transcript=[],
            previous_score=0.0,
            turn_count=1,
            max_turns=10,
            hidden_state="RR 28, O2 sat 82% at altitude",
        )

        assert input_data.student_action == "I check vitals"
        assert input_data.turn_count == 1
        assert input_data.previous_score == 0.0

    def test_score_bounds(self):
        """Test that previous_score is bounded 0.0-1.0."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            ActionResponseInput(
                student_action="Test",
                scenario_title="Test",
                scenario_setting="Test",
                patient_summary="Test",
                hidden_truth="Test",
                learning_objectives=["Test"],
                transcript=[],
                previous_score=1.5,  # Invalid: > 1.0
                turn_count=1,
                max_turns=5,
                hidden_state="Test",
            )

    def test_turn_count_bounds(self):
        """Test that turn_count must be >= 1."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            ActionResponseInput(
                student_action="Test",
                scenario_title="Test",
                scenario_setting="Test",
                patient_summary="Test",
                hidden_truth="Test",
                learning_objectives=["Test"],
                transcript=[],
                previous_score=0.0,
                turn_count=0,  # Invalid: < 1
                max_turns=5,
                hidden_state="Test",
            )
