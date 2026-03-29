# Judge System Implementation Plan

**Status:** Draft  
**Date:** 2026-03-29  
**Author:** Claude  
**Scope:** Multi-judge evaluation system for summit-sim using MLflow automatic evaluation

---

## Executive Summary

Implement a comprehensive judge evaluation system for the Action Responder agent's `DynamicTurnResult` outputs. The system uses MLflow's automatic evaluation with AI Gateway to assess simulation quality across multiple dimensions, providing actionable feedback for prompt improvement.

**Key Features:**
- 10 evaluation criteria across 4 judges
- Trace-level judges (structure, scoring, medical) evaluate every turn
- Session-level judge (continuity) evaluates on session completion
- Automatic rollup computation for optimization metric
- Fully asynchronous via MLflow automatic evaluation
- Configurable judge models to avoid bias

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Student Simulation Flow                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Action Turn                                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ Student Action  │───▶│ Action Responder│───▶│ DynamicTurn │ │
│  │   (free text)   │    │    (Agent)      │    │   Result    │ │
│  └─────────────────┘    └─────────────────┘    └──────┬──────┘ │
│                                                       │         │
│                              ┌────────────────────────┘         │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MLflow Automatic Evaluation                 │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐    │   │
│  │  │Structure    │ │   Scoring   │ │    Medical      │    │   │
│  │  │   Judge     │ │   Judge     │ │    Judge        │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘    │   │
│  │                                                         │   │
│  │  Criteria Evaluated (per turn):                         │   │
│  │  • score_in_range                    • score_milestone_justified  │
│  │  • question_in_narrative_only        • score_not_over_awarded    │
│  │  • feedback_no_harsh_language        • was_correct_treatment_gate │
│  │  • narrative_length                  • feedback_acknowledges_actions│
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Session Completion (5 min inactivity or explicit end)          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              MLflow Session Evaluation                     │ │
│  │  ┌─────────────────┐                                     │ │
│  │  │ Continuity Judge│                                     │ │
│  │  └─────────────────┘                                     │ │
│  │                                                          │ │
│  │  Criteria Evaluated (across all turns):                  │ │
│  │  • score_monotonic                                       │ │
│  │  • narrative_reveals_progressively                       │ │
│  │  • was_correct_treatment_gate (session-level)            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rollup Computation (application-side)                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Weighted Score = Σ(criterion_passed × weight)             │ │
│  │  Range: 0.0 - 1.0                                         │ │
│  │  Stored as metric: judge.overall_score                    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Criteria Specification

### Weight Distribution

| Criterion | Judge | Scope | Weight | Type |
|-----------|-------|-------|--------|------|
| score_in_range | Structure | Trace | 0.05 | bool |
| question_in_narrative_only | Structure | Trace | 0.10 | bool |
| feedback_no_harsh_language | Structure | Trace | 0.05 | bool |
| narrative_length | Structure | Trace | 0.05 | bool |
| score_milestone_justified | Scoring | Trace | 0.20 | bool |
| score_not_over_awarded | Scoring | Trace | 0.15 | bool |
| feedback_acknowledges_actions | Scoring | Trace | 0.05 | bool |
| was_correct_treatment_gate | Medical | Trace | 0.15 | bool |
| score_monotonic | Continuity | Session | 0.10 | bool |
| narrative_reveals_progressively | Continuity | Session | 0.10 | bool |

**Total Weights:**
- Trace-level: 0.80
- Session-level: 0.20

### Criterion Definitions

#### 1. score_in_range (weight: 0.05)
**Judge:** Structure  
**Scope:** Trace  
**Checks:** `completion_score` is between 0.0 and 1.0 inclusive  
**Rationale format:** "Score X is within valid range [0.0, 1.0]" or "Score X is out of bounds"

#### 2. question_in_narrative_only (weight: 0.10)
**Judge:** Structure  
**Scope:** Trace  
**Checks:** 
- `narrative_text` ends with an open question
- `feedback` contains NO questions whatsoever
**Rationale format:** "Narrative ends with question; feedback contains no questions" or "Feedback contains question at: '...'"

#### 3. feedback_no_harsh_language (weight: 0.05)
**Judge:** Structure  
**Scope:** Trace  
**Checks:** Feedback uses encouraging, constructive tone without harsh corrections
**Harsh language examples:** "STOP", "You should have...", "That's wrong", "Never do..."
**Rationale format:** "Feedback uses supportive tone" or "Harsh language detected: '...'"

#### 4. narrative_length (weight: 0.05)
**Judge:** Structure  
**Scope:** Trace  
**Checks:** `narrative_text` is 3-5 sentences
**Rationale format:** "Narrative is X sentences (valid)" or "Narrative is X sentences (expected 3-5)"

#### 5. score_milestone_justified (weight: 0.20)
**Judge:** Scoring  
**Scope:** Trace  
**Checks:** Score aligns with PAS rubric progress based on cumulative actions
**Reference rubric:**
- 0.00: Starting point
- 0.20: Scene size-up OR Primary assessment
- 0.40: Secondary assessment (vitals)
- 0.60: Treatment started
- 0.80: Extended care plan
- 1.00: Evacuation plan
**Rationale format:** "Score X aligns with completion of [milestone]" or "Score X seems high given only [actions]"

#### 6. score_not_over_awarded (weight: 0.15)
**Judge:** Scoring  
**Scope:** Trace  
**Checks:** Score jumps are reasonable (≤0.2 unless explicit bundling)
**Bundling exception:** If student explicitly states multiple assessment/treatment steps in one action, allow larger jumps
**Rationale format:** "Score increase of X is reasonable" or "Score jump of X exceeds threshold without clear bundling"

#### 7. feedback_acknowledges_actions (weight: 0.05)
**Judge:** Scoring  
**Scope:** Trace  
**Checks:** Feedback references the specific student action taken
**Valid:** "Good work checking vitals"  
**Invalid:** "Good work" (too generic)  
**Rationale format:** "Feedback mentions specific action: ..." or "Feedback lacks specific acknowledgment"

#### 8. was_correct_treatment_gate (weight: 0.15)
**Judge:** Medical  
**Scope:** Trace + Session  
**Checks:** `was_correct` accurately reflects whether student performed treatment without assessment
**Treat as incorrect:** Splinting, bandaging, medication, moving patient without vitals/head-to-toe  
**Treat as correct:** Vitals, physical exam, SAMPLE history, identification without intervention  
**Rationale format:** "was_correct=True is accurate; assessment without treatment" or "was_correct=True is wrong; treatment without assessment detected"

#### 9. score_monotonic (weight: 0.10)
**Judge:** Continuity  
**Scope:** Session  
**Checks:** `completion_score` never decreases across the session
**Rationale format:** "Scores are monotonically increasing: [list]" or "Score decreased from X to Y at turn Z"

#### 10. narrative_reveals_progressively (weight: 0.10)
**Judge:** Continuity  
**Scope:** Session  
**Checks:** No facts presented as "new discoveries" are repeated across narratives
**Exception:** Status updates ("HR is still 110") are allowed if framed as ongoing, not new  
**Rationale format:** "All facts revealed progressively" or "Repeated 'new' fact at turn X: '...'"

---

## Schema Additions

Add to `src/summit_sim/schemas.py`:

```python
class CriterionEvaluation(BaseModel):
    """Single criterion evaluation result."""
    name: str
    passed: bool
    weight: float
    reason: str


class TraceEvaluationResult(BaseModel):
    """Combined trace-level evaluation results."""
    trace_id: str
    session_id: str
    turn_number: int
    criteria: list[CriterionEvaluation]
    raw_score: float  # Sum of passed weights
    max_possible: float  # Sum of all weights


class SessionEvaluationResult(BaseModel):
    """Session-level evaluation results."""
    session_id: str
    total_turns: int
    criteria: list[CriterionEvaluation]
    raw_score: float
    max_possible: float


class RollupMetric(BaseModel):
    """Final weighted score for optimization."""
    session_id: str
    overall_score: float  # 0.0 - 1.0
    trace_contribution: float  # 0.0 - 0.8
    session_contribution: float  # 0.0 - 0.2
    breakdown: dict[str, bool]  # criterion_name -> passed
```

---

## File Structure

```
src/summit_sim/
├── judges/
│   ├── __init__.py           # Public API exports
│   ├── config.py             # Judge configuration and weights
│   ├── schemas.py            # Judge-specific schemas (optional, can use main schemas.py)
│   └── setup.py              # Judge registration and initialization
├── schemas.py                # Add evaluation schemas
├── graphs/
│   └── simulation.py         # Add trace metadata for judges
└── ui/
    └── simulation.py         # Trigger session completion signal
```

---

## Judge Prompts

### Trace-Level Judges

#### Structure Judge

```python
STRUCTURE_JUDGE_INSTRUCTIONS = """
You are evaluating the structure and formatting of an AI-generated response in a wilderness first responder training simulation.

Evaluate the following output against these criteria:

1. SCORE_IN_RANGE: Is completion_score between 0.0 and 1.0?
2. QUESTION_IN_NARRATIVE_ONLY: Does narrative_text end with an open question? Does feedback contain NO questions?
3. FEEDBACK_NO_HARSH_LANGUAGE: Is the feedback encouraging and constructive without harsh corrections?
4. NARRATIVE_LENGTH: Is narrative_text between 3-5 sentences?

Output format (JSON):
{
    "score_in_range": {"passed": bool, "reason": "1-2 sentences"},
    "question_in_narrative_only": {"passed": bool, "reason": "1-2 sentences"},
    "feedback_no_harsh_language": {"passed": bool, "reason": "1-2 sentences"},
    "narrative_length": {"passed": bool, "reason": "1-2 sentences"}
}

Be concise. Cite exact mismatches when criteria fail.
"""
```

#### Scoring Judge

```python
SCORING_JUDGE_INSTRUCTIONS = """
You are evaluating the scoring accuracy and pedagogical quality of an AI-generated response in a wilderness first responder training simulation.

Context:
- PAS Rubric: 0.0=start, 0.2=scene/primary, 0.4=secondary, 0.6=treatment, 0.8=extended care, 1.0=evacuation
- Student transcript shows all previous actions

Evaluate:
1. SCORE_MILESTONE_JUSTIFIED: Does completion_score align with the PAS rubric based on cumulative actions?
2. SCORE_NOT_OVER_AWARDED: Is the score increase from previous turn reasonable (≤0.2 unless explicit bundling)?
3. FEEDBACK_ACKNOWLEDGES_ACTIONS: Does feedback specifically mention the student's action?

Output format (JSON):
{
    "score_milestone_justified": {"passed": bool, "reason": "1-2 sentences"},
    "score_not_over_awarded": {"passed": bool, "reason": "1-2 sentences"},
    "feedback_acknowledges_actions": {"passed": bool, "reason": "1-2 sentences"}
}

Be concise. Cite specific actions and score values.
"""
```

#### Medical Judge

```python
MEDICAL_JUDGE_INSTRUCTIONS = """
You are evaluating the medical accuracy of an AI-generated response in a wilderness first responder training simulation.

Evaluate:
WAS_CORRECT_TREATMENT_GATE: Is was_correct accurate?
- was_correct should be FALSE if student performed treatment (splint, bandage, medication, move patient) without assessment
- was_correct should be TRUE for assessment actions (vitals, exam, SAMPLE history)

Reference hidden_truth and hidden_state to determine if treatment was premature.

Output format (JSON):
{
    "was_correct_treatment_gate": {"passed": bool, "reason": "1-2 sentences citing specific action"}
}
"""
```

### Session-Level Judge

```python
CONTINUITY_JUDGE_INSTRUCTIONS = """
You are evaluating the continuity and progression across a complete wilderness first responder simulation session.

You have access to all traces in the session. Evaluate:

1. SCORE_MONOTONIC: Does completion_score never decrease across turns?
2. NARRATIVE_REVEALS_PROGRESSIVELY: Are facts presented as "new discoveries" never repeated? (Status updates like "HR is still 110" are OK if framed as ongoing)
3. WAS_CORRECT_TREATMENT_GATE: Across the session, was was_correct consistently accurate?

Output format (JSON):
{
    "score_monotonic": {"passed": bool, "reason": "1-2 sentences"},
    "narrative_reveals_progressively": {"passed": bool, "reason": "1-2 sentences"},
    "was_correct_treatment_gate": {"passed": bool, "reason": "1-2 sentences"}
}

Be concise. Cite specific turns where issues occur.
"""
```

---

## MLflow AI Gateway Configuration

### Prerequisites

1. MLflow Server running with AI Gateway enabled
2. AI Gateway endpoints configured for judge models

### Recommended Endpoints

Create these endpoints in MLflow AI Gateway UI:

1. **judge-structure** 
   - Model: anthropic/claude-3-5-sonnet-20241022 or google/gemini-2.0-flash-lite
   - Purpose: Structure judge

2. **judge-scoring**
   - Model: anthropic/claude-3-5-sonnet-20241022 or google/gemini-2.0-flash-lite  
   - Purpose: Scoring judge

3. **judge-medical**
   - Model: anthropic/claude-3-5-sonnet-20241022 or google/gemini-2.0-flash-lite
   - Purpose: Medical judge

4. **judge-continuity**
   - Model: anthropic/claude-3-5-sonnet-20241022 or google/gemini-2.0-flash-lite
   - Purpose: Session-level continuity judge

**Note:** Use a different model family than action_responder (gemini) to avoid bias.

---

## Implementation Steps

### Phase 1: Configuration and Setup

**File:** `src/summit_sim/judges/config.py`

```python
"""Judge configuration and weights."""

from typing import Literal

# Judge weights (must sum to 1.0 across all criteria)
JUDGE_WEIGHTS: dict[str, float] = {
    # Trace-level (Structure Judge)
    "score_in_range": 0.05,
    "question_in_narrative_only": 0.10,
    "feedback_no_harsh_language": 0.05,
    "narrative_length": 0.05,
    # Trace-level (Scoring Judge)
    "score_milestone_justified": 0.20,
    "score_not_over_awarded": 0.15,
    "feedback_acknowledges_actions": 0.05,
    # Trace-level (Medical Judge)
    "was_correct_treatment_gate_trace": 0.15,
    # Session-level (Continuity Judge)
    "score_monotonic": 0.10,
    "narrative_reveals_progressively": 0.10,
    "was_correct_treatment_gate_session": 0.10,
}

# Model endpoint configuration
JUDGE_MODELS: dict[str, str] = {
    "structure": "gateway:/judge-structure",
    "scoring": "gateway:/judge-scoring",
    "medical": "gateway:/judge-medical",
    "continuity": "gateway:/judge-continuity",
}

# Sampling configuration
TRACE_JUDGE_SAMPLE_RATE: float = 1.0  # 100% in dev
SESSION_JUDGE_SAMPLE_RATE: float = 1.0  # 100% in dev
```

### Phase 2: Judge Registration

**File:** `src/summit_sim/judges/setup.py`

```python
"""Judge registration and initialization."""

import logging

import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import ScorerSamplingConfig

from summit_sim.judges.config import JUDGE_MODELS

logger = logging.getLogger(__name__)

# Judge instances (created on first use)
_judges: dict[str, any] = {}


def get_structure_judge():
    """Get or create structure judge."""
    if "structure" not in _judges:
        _judges["structure"] = make_judge(
            name="structure-judge",
            instructions=STRUCTURE_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODELS["structure"],
            feedback_value_type=dict,  # Returns JSON with criteria
        )
    return _judges["structure"]


def get_scoring_judge():
    """Get or create scoring judge."""
    if "scoring" not in _judges:
        _judges["scoring"] = make_judge(
            name="scoring-judge",
            instructions=SCORING_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODELS["scoring"],
            feedback_value_type=dict,
        )
    return _judges["scoring"]


def get_medical_judge():
    """Get or create medical judge."""
    if "medical" not in _judges:
        _judges["medical"] = make_judge(
            name="medical-judge",
            instructions=MEDICAL_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODELS["medical"],
            feedback_value_type=dict,
        )
    return _judges["medical"]


def get_continuity_judge():
    """Get or create continuity judge."""
    if "continuity" not in _judges:
        _judges["continuity"] = make_judge(
            name="continuity-judge",
            instructions=CONTINUITY_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODELS["continuity"],
            feedback_value_type=dict,
        )
    return _judges["continuity"]


def register_trace_judges():
    """Register trace-level judges for automatic evaluation."""
    experiment = mlflow.get_experiment_by_name("summit-sim-judges")
    if not experiment:
        experiment_id = mlflow.create_experiment("summit-sim-judges")
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_id)
    
    # Register structure judge
    structure = get_structure_judge()
    registered = structure.register(name="trace-structure")
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
        filter_string="tags.graph_type = 'simulation' AND tags.agent_name = 'action-responder'"
    )
    logger.info("Registered structure judge for automatic evaluation")
    
    # Register scoring judge
    scoring = get_scoring_judge()
    registered = scoring.register(name="trace-scoring")
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
        filter_string="tags.graph_type = 'simulation' AND tags.agent_name = 'action-responder'"
    )
    logger.info("Registered scoring judge for automatic evaluation")
    
    # Register medical judge
    medical = get_medical_judge()
    registered = medical.register(name="trace-medical")
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
        filter_string="tags.graph_type = 'simulation' AND tags.agent_name = 'action-responder'"
    )
    logger.info("Registered medical judge for automatic evaluation")


def register_session_judges():
    """Register session-level judges for automatic evaluation."""
    experiment = mlflow.get_experiment_by_name("summit-sim-judges")
    mlflow.set_experiment(experiment.experiment_id)
    
    # Register continuity judge (session scope)
    continuity = get_continuity_judge()
    registered = continuity.register(name="session-continuity")
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
        # Session judges automatically target sessions, not individual traces
    )
    logger.info("Registered continuity judge for automatic evaluation")
```

### Phase 3: Trace Metadata Enhancement

**File:** `src/summit_sim/graphs/simulation.py`

Update the `process_player_action` node to include judge-relevant metadata:

```python
@mlflow.trace(span_type=SpanType.AGENT)
async def process_player_action(state: SimulationState, config: RunnableConfig) -> dict:
    """Process player action and generate next turn."""
    # ... existing code ...
    
    result = await process_action(...)
    
    # Add judge-relevant span attributes
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_attributes({
            "simulation.turn_count": state.turn_count + 1,
            "agent.name": ACTION_RESPONDER_AGENT_NAME,
            "turn.was_correct": result.was_correct,
            "turn.completion_score": result.completion_score,
            "turn.previous_score": previous_score,
            "turn.score_delta": result.completion_score - previous_score,
            "turn.student_action": student_action[:100],  # Truncated for context
            "turn.narrative_length": len(result.narrative_text.split('.')),
            "judge.context.hidden_truth": state.hidden_truth[:200],  # For medical judge
            "judge.context.transcript": _format_transcript_for_judge(state.transcript),
        })
    
    return {...}


def _format_transcript_for_judge(transcript: list[TranscriptEntry]) -> str:
    """Format transcript for judge context."""
    lines = []
    for entry in transcript:
        lines.append(f"Turn {entry.turn_id}: {entry.student_action[:80]}...")
        lines.append(f"  Score: {entry.score}, Correct: {entry.was_correct}")
    return "\n".join(lines)
```

### Phase 4: Rollup Computation

**File:** `src/summit_sim/judges/rollup.py`

```python
"""Rollup computation for overall quality score."""

import logging
from typing import Any

import mlflow

from summit_sim.judges.config import JUDGE_WEIGHTS

logger = logging.getLogger(__name__)


def compute_rollup_score(session_id: str) -> dict[str, Any]:
    """Compute overall score from all judge assessments in a session.
    
    This should be called after session completion.
    """
    # Query MLflow for all assessments in this session
    client = mlflow.tracking.MlflowClient()
    
    # Get traces for this session
    traces = client.search_traces(
        experiment_ids=[mlflow.get_experiment_by_name("summit-sim-judges").experiment_id],
        filter_string=f"tags.session_id = '{session_id}'",
    )
    
    # Aggregate criterion results
    criterion_results: dict[str, bool] = {}
    total_score = 0.0
    
    for trace in traces:
        # Get assessments for this trace
        assessments = client.get_trace(trace.info.trace_id).assessments
        
        for assessment in assessments:
            criterion_name = assessment.name
            passed = assessment.value  # Assuming boolean
            
            if criterion_name in JUDGE_WEIGHTS:
                criterion_results[criterion_name] = passed
                if passed:
                    total_score += JUDGE_WEIGHTS[criterion_name]
    
    overall_score = total_score  # Already weighted
    
    # Log as metric
    mlflow.log_metric(
        key="judge.overall_score",
        value=overall_score,
        run_id=session_id,  # Or appropriate run ID
    )
    
    return {
        "session_id": session_id,
        "overall_score": overall_score,
        "breakdown": criterion_results,
        "total_criteria": len(criterion_results),
        "passed_criteria": sum(1 for v in criterion_results.values() if v),
    }
```

### Phase 5: Initialization

**File:** `src/summit_sim/main.py` (or appropriate startup location)

```python
# In application startup
def initialize_judges():
    """Initialize automatic evaluation judges."""
    from summit_sim.judges.setup import register_trace_judges, register_session_judges
    
    register_trace_judges()
    register_session_judges()
    logger.info("Automatic evaluation judges registered")


# Call during app startup
initialize_judges()
```

---

## Testing Strategy

### Unit Tests

Create `tests/judges/test_judges.py`:

```python
"""Tests for judge evaluation logic."""

import pytest
from unittest.mock import AsyncMock, patch

from summit_sim.judges.rollup import compute_rollup_score
from summit_sim.schemas import DynamicTurnResult


@pytest.fixture
def sample_turn_result():
    return DynamicTurnResult(
        was_correct=True,
        completion_score=0.4,
        feedback="Good work checking vitals!",
        narrative_text="You check pulse and breathing. Patient responds normally. What do you check next?",
    )


def test_compute_rollup_score_weights():
    """Test that rollup respects weights."""
    # Mock MLflow client
    with patch("mlflow.tracking.MlflowClient") as mock_client:
        # Set up mock assessments
        mock_assessments = [
            {"name": "score_milestone_justified", "value": True},  # 0.20
            {"name": "was_correct_treatment_gate", "value": True},  # 0.15
            {"name": "score_monotonic", "value": False},  # 0.10 (failed)
        ]
        
        result = compute_rollup_score("test-session")
        
        # Should be 0.35 (0.20 + 0.15)
        assert result["overall_score"] == 0.35
```

### Integration Tests

Test judge registration and MLflow integration with mocked LLM calls.

---

## Deployment Checklist

- [ ] Set up AI Gateway endpoints for all 4 judge models
- [ ] Create MLflow experiment "summit-sim-judges"
- [ ] Add `JUDGE_MODELS` configuration (via env vars or config)
- [ ] Deploy updated application with judge registration
- [ ] Verify judges appear in MLflow UI under experiment Judges tab
- [ ] Test with sample simulation session
- [ ] Verify assessments appear in traces
- [ ] Verify rollup computation on session completion
- [ ] Adjust sampling rates for production (if needed)

---

## Monitoring and Debugging

### Viewing Results

1. **MLflow UI**: Navigate to experiment "summit-sim-judges"
2. **Traces Tab**: See individual criterion assessments as columns
3. **Overview Tab**: View trend charts for overall_score and individual criteria
4. **Filter**: Use `tags.session_id = 'xxx'` to view specific sessions

### Troubleshooting

**Issue:** Judges not evaluating
- Check AI Gateway endpoints are configured
- Verify filter strings match trace tags
- Check MLflow server logs for evaluation errors

**Issue:** Session judge not running
- Sessions complete after 5 min of inactivity (configurable via `MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS`)
- Or trigger explicit completion signal

**Issue:** Weights don't sum correctly
- Run validation: `sum(JUDGE_WEIGHTS.values())` should equal 1.0

---

## Future Enhancements

1. **Judge Alignment**: Use `judge.align()` after collecting human feedback examples
2. **Custom Filters**: Add production filters to focus evaluation on high-risk traces
3. **Alerting**: Set up alerts when overall_score drops below threshold
4. **A/B Testing**: Compare judge scores across different prompt versions
5. **Golden Scenarios**: Store high-scoring sessions as reference for judge training

---

## References

- MLflow Automatic Evaluation: https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/
- MLflow Judges: https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/
- MLflow AI Gateway: https://mlflow.org/docs/latest/genai/governance/ai-gateway/
- Summit-Sim Architecture: `plans/high-level-arch.md`
