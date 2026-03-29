# Judge System Implementation Plan

**Status:** Draft  
**Date:** 2026-03-29  
**Author:** Claude  
**Scope:** Multi-judge evaluation system for summit-sim using MLflow automatic evaluation

---

## Executive Summary

Implement a comprehensive judge evaluation system for the Action Responder agent's `DynamicTurnResult` outputs. The system uses MLflow's automatic evaluation with AI Gateway to assess simulation quality across multiple dimensions, providing actionable feedback for prompt improvement.

**Key Features:**
- 9 evaluation criteria across 4 judges (reduced from 10 - removed duplicate)
- Trace-level judges (structure, scoring, medical) evaluate every turn within 1-2 minutes
- Session-level judge (continuity) evaluates on session completion (5 min inactivity)
- Offline rollup computation via notebook for optimization metric
- Fully asynchronous via MLflow automatic evaluation
- Single gateway endpoint for all judges (configurable model via env var)

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
│  │  Criteria Evaluated (per turn, within 1-2 min):         │   │
│  │  • score_in_range                    • score_milestone_justified  │
│  │  • question_in_narrative_only        • score_not_over_awarded    │
│  │  • feedback_no_harsh_language        • was_correct_treatment_gate │
│  │  • narrative_length                  • feedback_acknowledges_actions│
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Session Completion (5 min inactivity)                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              MLflow Session Evaluation                     │ │
│  │  ┌─────────────────┐                                     │ │
│  │  │ Continuity Judge│                                     │ │
│  │  └─────────────────┘                                     │ │
│  │                                                          │ │
│  │  Criteria Evaluated (across all turns):                  │ │
│  │  • score_monotonic                                       │ │
│  │  • narrative_reveals_progressively                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rollup Computation (offline in notebook)                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Weighted Score = Σ(criterion_passed × weight)             │ │
│  │  Range: 0.0 - 1.0                                         │ │
│  │  Run after traces/sessions are evaluated                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Timing

MLflow automatic evaluation operates asynchronously with the following timing:

| Evaluation Type | When It Runs | Timing |
|-----------------|--------------|--------|
| **Trace-level** | After each trace is logged | 1-2 minutes |
| **Session-level** | After session completion | 5 minutes of inactivity (configurable via `MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS`) |

**Key Implications:**
- Trace assessments appear within minutes of simulation turns completing
- Session assessments appear 5 minutes after the last trace in a session
- This enables **offline rollup computation** - run `compute_rollup_score()` in a Jupyter notebook after simulations complete
- No need for real-time rollup triggers in the application code

---

## Evaluation Criteria Specification

### Weight Distribution

| Criterion | Judge | Scope | Weight | Type |
|-----------|-------|-------|--------|------|
| score_in_range | Structure | Trace | 0.05 | bool |
| question_in_narrative_only | Structure | Trace | 0.10 | bool |
| feedback_no_harsh_language | Structure | Trace | 0.05 | bool |
| narrative_length | Structure | Trace | 0.05 | bool |
| score_milestone_justified | Scoring | Trace | 0.25 | bool |
| score_not_over_awarded | Scoring | Trace | 0.15 | bool |
| feedback_acknowledges_actions | Scoring | Trace | 0.05 | bool |
| was_correct_treatment_gate | Medical | Trace | 0.15 | bool |
| score_monotonic | Continuity | Session | 0.10 | bool |
| narrative_reveals_progressively | Continuity | Session | 0.10 | bool |

**Total Weights: 1.00**
- Trace-level: 0.85
- Session-level: 0.20

**Changes from v1:**
- Removed duplicate `was_correct_treatment_gate_session` (was 0.10)
- Increased `score_milestone_justified` from 0.20 to 0.25 (most important criterion)
- `was_correct_treatment_gate` now evaluated only at trace level (0.15)

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

#### 5. score_milestone_justified (weight: 0.25)
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
**Scope:** Trace  
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

## Schema Design

Per MLflow best practices, each judge has a **focused, specific output schema** rather than a unified schema. This enables reusability and maintainability.

### Judge Output Schemas

Each judge returns a **single assessment** per evaluation:

```python
# Structure Judge - returns dict with 4 criteria
StructureJudgeOutput = dict[str, dict[str, str | bool]]  # {criterion: {passed: bool, reason: str}}

# Scoring Judge - returns dict with 3 criteria  
ScoringJudgeOutput = dict[str, dict[str, str | bool]]

# Medical Judge - returns bool (single criterion)
MedicalJudgeOutput = bool

# Continuity Judge - returns dict with 2 criteria
ContinuityJudgeOutput = dict[str, dict[str, str | bool]]
```

### Rollup Schema (for offline computation)

```python
class RollupResult(BaseModel):
    """Final weighted score for optimization."""
    session_id: str
    overall_score: float  # 0.0 - 1.0
    trace_contribution: float  # 0.0 - 0.85
    session_contribution: float  # 0.0 - 0.20
    breakdown: dict[str, bool]  # criterion_name -> passed
    total_criteria: int
    passed_criteria: int
```

**Design Rationale:**
- MLflow recommends "multiple judges for comprehensive quality coverage"
- Built-in judges follow this pattern (Correctness, Safety, RelevanceToQuery are all separate)
- Individual judges can be reused across different agents/scenarios
- Easier to update one criterion without affecting others

---

## File Structure

```
src/summit_sim/
├── judges/
│   ├── __init__.py           # Public API exports
│   ├── config.py             # Judge configuration, weights, prompts
│   ├── setup.py              # Judge registration and initialization
│   └── rollup.py             # Offline rollup computation
├── schemas.py                # Add RollupResult schema
├── graphs/
│   └── simulation.py         # Trace metadata for judges (already has session_id tagging)
└── notebooks/
    └── judge_analysis.ipynb  # Offline rollup and analysis
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

Output format: Return True if was_correct is accurate, False otherwise.
"""
```

### Session-Level Judge

```python
CONTINUITY_JUDGE_INSTRUCTIONS = """
You are evaluating the continuity and progression across a complete wilderness first responder simulation session.

You have access to all traces in the session. Evaluate:

1. SCORE_MONOTONIC: Does completion_score never decrease across turns?
2. NARRATIVE_REVEALS_PROGRESSIVELY: Are facts presented as "new discoveries" never repeated? (Status updates like "HR is still 110" are OK if framed as ongoing)

Output format (JSON):
{
    "score_monotonic": {"passed": bool, "reason": "1-2 sentences"},
    "narrative_reveals_progressively": {"passed": bool, "reason": "1-2 sentences"}
}

Be concise. Cite specific turns where issues occur.
"""
```

---

## MLflow AI Gateway Configuration

### Prerequisites

1. MLflow Server running with AI Gateway enabled
2. AI Gateway endpoint configured for judge model

### Configuration

**Single Endpoint for All Judges:**

Create one endpoint in MLflow AI Gateway UI:

- **Name:** `judge-model`
- **Model:** Configurable via environment variable (default: `anthropic/claude-3-5-sonnet-20241022`)
- **Purpose:** All judges (structure, scoring, medical, continuity)

**Environment Variable:**
```bash
JUDGE_MODEL="anthropic/claude-3-5-sonnet-20241022"  # or google/gemini-2.0-flash-lite
```

**Note:** Use a different model family than action_responder (gemini) to avoid bias.

---

## Implementation Steps

### Phase 1: Configuration

**File:** `src/summit_sim/judges/config.py`

```python
"""Judge configuration, prompts, and weights."""

import os

# Judge weights (must sum to 1.0 across all criteria)
JUDGE_WEIGHTS: dict[str, float] = {
    # Trace-level (Structure Judge)
    "score_in_range": 0.05,
    "question_in_narrative_only": 0.10,
    "feedback_no_harsh_language": 0.05,
    "narrative_length": 0.05,
    # Trace-level (Scoring Judge)
    "score_milestone_justified": 0.25,  # Increased from 0.20
    "score_not_over_awarded": 0.15,
    "feedback_acknowledges_actions": 0.05,
    # Trace-level (Medical Judge)
    "was_correct_treatment_gate": 0.15,
    # Session-level (Continuity Judge)
    "score_monotonic": 0.10,
    "narrative_reveals_progressively": 0.10,
}

# Validate weights sum to 1.0
assert abs(sum(JUDGE_WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"

# Single gateway endpoint for all judges
JUDGE_MODEL_ENDPOINT = f"gateway:/{os.getenv('JUDGE_MODEL', 'judge-model')}"

# Sampling configuration
TRACE_JUDGE_SAMPLE_RATE: float = 1.0  # 100% in dev
SESSION_JUDGE_SAMPLE_RATE: float = 1.0  # 100% in dev

# Prompts (defined above)
STRUCTURE_JUDGE_INSTRUCTIONS = """..."""
SCORING_JUDGE_INSTRUCTIONS = """..."""
MEDICAL_JUDGE_INSTRUCTIONS = """..."""
CONTINUITY_JUDGE_INSTRUCTIONS = """..."""
```

### Phase 2: Judge Registration

**File:** `src/summit_sim/judges/setup.py`

```python
"""Judge registration and initialization."""

import logging

import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import ScorerSamplingConfig

from summit_sim.judges.config import (
    CONTINUITY_JUDGE_INSTRUCTIONS,
    JUDGE_MODEL_ENDPOINT,
    MEDICAL_JUDGE_INSTRUCTIONS,
    SCORING_JUDGE_INSTRUCTIONS,
    STRUCTURE_JUDGE_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

# Judge instances (created on first use)
_judges: dict[str, any] = {}


def get_structure_judge():
    """Get or create structure judge."""
    if "structure" not in _judges:
        _judges["structure"] = make_judge(
            name="structure-judge",
            instructions=STRUCTURE_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=dict,
        )
    return _judges["structure"]


def get_scoring_judge():
    """Get or create scoring judge."""
    if "scoring" not in _judges:
        _judges["scoring"] = make_judge(
            name="scoring-judge",
            instructions=SCORING_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=dict,
        )
    return _judges["scoring"]


def get_medical_judge():
    """Get or create medical judge."""
    if "medical" not in _judges:
        _judges["medical"] = make_judge(
            name="medical-judge",
            instructions=MEDICAL_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
            feedback_value_type=bool,
        )
    return _judges["medical"]


def get_continuity_judge():
    """Get or create continuity judge."""
    if "continuity" not in _judges:
        _judges["continuity"] = make_judge(
            name="continuity-judge",
            instructions=CONTINUITY_JUDGE_INSTRUCTIONS,
            model=JUDGE_MODEL_ENDPOINT,
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
    
    # Register all trace-level judges
    judges = [
        ("structure", get_structure_judge()),
        ("scoring", get_scoring_judge()),
        ("medical", get_medical_judge()),
    ]
    
    for name, judge in judges:
        registered = judge.register(name=f"trace-{name}")
        registered.start(
            sampling_config=ScorerSamplingConfig(sample_rate=1.0),
            filter_string="tags.graph_type = 'simulation' AND tags.agent_name = 'action-responder'"
        )
        logger.info(f"Registered {name} judge for automatic evaluation")


def register_session_judges():
    """Register session-level judges for automatic evaluation."""
    experiment = mlflow.get_experiment_by_name("summit-sim-judges")
    mlflow.set_experiment(experiment.experiment_id)
    
    # Register continuity judge (session scope)
    continuity = get_continuity_judge()
    registered = continuity.register(name="session-continuity")
    registered.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
        # Session judges automatically target sessions via mlflow.trace.session metadata
    )
    logger.info("Registered continuity judge for automatic evaluation")


def initialize_judges():
    """Initialize all judges. Call during app startup."""
    register_trace_judges()
    register_session_judges()
    logger.info("Automatic evaluation judges registered")
```

### Phase 3: Trace Metadata Enhancement

**File:** `src/summit_sim/graphs/simulation.py`

The current implementation already tags traces with session_id. Ensure the following are set:

```python
# Already present in current code (lines 167-177):
mlflow.update_current_trace(
    metadata={"mlflow.trace.session": thread_id},
    tags={
        "session_id": thread_id,
        "scenario_id": state.scenario_id,
        "graph_type": "simulation",
        "mlflow_env": settings.mlflow_env,
    },
)
```

The trace metadata should already be sufficient for judges. The session_id tag enables session-level evaluation.

### Phase 4: Rollup Computation (Offline)

**File:** `src/summit_sim/judges/rollup.py`

```python
"""Rollup computation for overall quality score.

This module is designed to be used offline in a Jupyter notebook after
simulations complete. MLflow automatic evaluation takes 1-2 minutes for
traces and 5 minutes for sessions, so run this after waiting for evaluation.
"""

import logging
from typing import Any

import mlflow

from summit_sim.judges.config import JUDGE_WEIGHTS
from summit_sim.schemas import RollupResult

logger = logging.getLogger(__name__)


def compute_rollup_score(session_id: str) -> RollupResult:
    """Compute overall score from all judge assessments in a session.
    
    Run this in a Jupyter notebook after simulations complete and MLflow
    has had time to evaluate traces (1-2 min) and sessions (5 min inactivity).
    
    Args:
        session_id: The session ID to compute rollup for
        
    Returns:
        RollupResult with overall score and breakdown
    """
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
        trace_data = client.get_trace(trace.info.trace_id)
        if hasattr(trace_data, 'assessments'):
            for assessment in trace_data.assessments:
                criterion_name = assessment.name
                passed = assessment.value
                
                if criterion_name in JUDGE_WEIGHTS:
                    criterion_results[criterion_name] = passed
                    if passed:
                        total_score += JUDGE_WEIGHTS[criterion_name]
    
    # Calculate contributions
    trace_criteria = {k: v for k, v in criterion_results.items() 
                     if k not in ["score_monotonic", "narrative_reveals_progressively"]}
    session_criteria = {k: v for k, v in criterion_results.items() 
                       if k in ["score_monotonic", "narrative_reveals_progressively"]}
    
    trace_contribution = sum(JUDGE_WEIGHTS[k] for k, v in trace_criteria.items() if v)
    session_contribution = sum(JUDGE_WEIGHTS[k] for k, v in session_criteria.items() if v)
    
    return RollupResult(
        session_id=session_id,
        overall_score=total_score,
        trace_contribution=trace_contribution,
        session_contribution=session_contribution,
        breakdown=criterion_results,
        total_criteria=len(criterion_results),
        passed_criteria=sum(1 for v in criterion_results.values() if v),
    )


def compute_rollup_for_all_sessions(experiment_name: str = "summit-sim-judges") -> list[RollupResult]:
    """Compute rollup scores for all sessions in an experiment.
    
    Use this in a notebook to batch-process all completed simulations.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logger.warning(f"Experiment {experiment_name} not found")
        return []
    
    # Get all unique session IDs
    traces = client.search_traces(experiment_ids=[experiment.experiment_id])
    session_ids = set()
    for trace in traces:
        trace_data = client.get_trace(trace.info.trace_id)
        if hasattr(trace_data, 'tags'):
            session_id = trace_data.tags.get("session_id")
            if session_id:
                session_ids.add(session_id)
    
    # Compute rollup for each session
    results = []
    for session_id in session_ids:
        try:
            rollup = compute_rollup_score(session_id)
            results.append(rollup)
        except Exception as e:
            logger.error(f"Failed to compute rollup for session {session_id}: {e}")
    
    return results
```

### Phase 5: Initialization

**File:** `src/summit_sim/main.py`

```python
# In application startup
def initialize_judges():
    """Initialize automatic evaluation judges."""
    from summit_sim.judges.setup import initialize_judges as _init_judges
    
    _init_judges()


# Call during app startup
initialize_judges()
```

---

## Notebook Usage

**File:** `notebooks/judge_analysis.ipynb`

```python
# After running simulations, wait for automatic evaluation:
# - Trace assessments: 1-2 minutes
# - Session assessments: 5 minutes after last trace

from summit_sim.judges.rollup import compute_rollup_for_all_sessions

# Compute rollup for all sessions
results = compute_rollup_for_all_sessions("summit-sim-judges")

# Analyze results
for r in results:
    print(f"Session {r.session_id}: {r.overall_score:.2f}")
    print(f"  Passed: {r.passed_criteria}/{r.total_criteria}")
    
# Find sessions with low scores for investigation
low_score_sessions = [r for r in results if r.overall_score < 0.7]
```

---

## Testing Strategy

### Unit Tests

Create `tests/judges/test_rollup.py`:

```python
"""Tests for rollup computation logic."""

from unittest.mock import MagicMock, patch

from summit_sim.judges.rollup import compute_rollup_score
from summit_sim.judges.config import JUDGE_WEIGHTS


def test_compute_rollup_score_weights():
    """Test that rollup respects weights."""
    # Mock MLflow client
    mock_trace = MagicMock()
    mock_trace.info.trace_id = "trace-1"
    
    mock_assessment_struct = MagicMock()
    mock_assessment_struct.name = "score_milestone_justified"
    mock_assessment_struct.value = True
    
    mock_assessment_med = MagicMock()
    mock_assessment_med.name = "was_correct_treatment_gate"
    mock_assessment_med.value = True
    
    mock_assessment_cont = MagicMock()
    mock_assessment_cont.name = "score_monotonic"
    mock_assessment_cont.value = False
    
    mock_trace_data = MagicMock()
    mock_trace_data.assessments = [
        mock_assessment_struct,
        mock_assessment_med,
        mock_assessment_cont,
    ]
    
    with patch("mlflow.tracking.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.search_traces.return_value = [mock_trace]
        mock_client.get_trace.return_value = mock_trace_data
        
        result = compute_rollup_score("test-session")
        
        # Should be 0.40 (0.25 + 0.15)
        assert result.overall_score == 0.40
        assert result.passed_criteria == 2


def test_weight_sum_validation():
    """Test that weights sum to 1.0."""
    total = sum(JUDGE_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"
```

### Integration Tests

Test judge registration and MLflow integration with mocked LLM calls.

---

## Deployment Checklist

- [ ] Set up AI Gateway endpoint `judge-model` (or configure `JUDGE_MODEL` env var)
- [ ] Create MLflow experiment "summit-sim-judges"
- [ ] Set `JUDGE_MODEL` environment variable (optional, defaults to `judge-model`)
- [ ] Deploy updated application with judge registration
- [ ] Verify judges appear in MLflow UI under experiment Judges tab
- [ ] Test with sample simulation session
- [ ] Verify assessments appear in traces (wait 1-2 minutes)
- [ ] Verify session assessments appear after 5 min inactivity
- [ ] Run notebook to verify rollup computation works offline
- [ ] Adjust sampling rates for production (if needed)

---

## Monitoring and Debugging

### Viewing Results

1. **MLflow UI**: Navigate to experiment "summit-sim-judges"
2. **Traces Tab**: See individual criterion assessments as columns (appears within 1-2 min)
3. **Overview Tab**: View trend charts for overall_score and individual criteria
4. **Filter**: Use `tags.session_id = 'xxx'` to view specific sessions

### Troubleshooting

**Issue:** Judges not evaluating
- Check AI Gateway endpoint is configured
- Verify filter strings match trace tags
- Check MLflow server logs for evaluation errors
- Ensure traces are less than 1 hour old (lookback window)

**Issue:** Session judge not running
- Sessions complete after 5 min of inactivity (configurable via `MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS`)
- If new traces are added after evaluation, the session will be re-evaluated

**Issue:** Rollup computation returns empty results
- Wait for automatic evaluation to complete (traces: 1-2 min, sessions: 5 min)
- Verify session_id tags are present in traces
- Check that assessments exist in MLflow UI first

**Issue:** Weights don't sum correctly
- Run validation: `sum(JUDGE_WEIGHTS.values())` should equal 1.0
- Check that `was_correct_treatment_gate_session` was removed

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
- MLflow Session Evaluation Timing: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/multi-turn/
- Summit-Sim Architecture: `plans/high-level-arch.md`

---

## Changelog

### v2 (2026-03-29)
- **Removed**: Duplicate `was_correct_treatment_gate_session` criterion
- **Changed**: `score_milestone_justified` weight from 0.20 to 0.25
- **Changed**: Single gateway endpoint for all judges (configurable via `JUDGE_MODEL` env var)
- **Changed**: Rollup computation to offline notebook workflow
- **Added**: Clarified evaluation timing (1-2 min for traces, 5 min for sessions)
- **Validated**: Multiple focused judges is MLflow best practice
- **Fixed**: Weights now sum to exactly 1.0
