# Judge System Implementation

**Status:** ✅ Implemented (Disabled Due to MLflow Bug)  
**Date:** 2026-03-29  
**MLflow Bug:** [#20782](https://github.com/mlflow/mlflow/issues/20782) - Automated scoring fails with S3 artifact storage  
**Fix PR:** [#20784](https://github.com/mlflow/mlflow/pull/20784) - Not yet merged

---

## Summary

The judge evaluation system is fully implemented and ready to use, but **automated MLflow evaluation is disabled** due to a bug in MLflow 3.10.1 that prevents automated scoring from working with S3 artifact storage.

**Current State:**
- ✅ All judge code implemented and tested (102 tests pass)
- ✅ Manual/offline rollup computation works via notebooks
- ❌ Automated evaluation disabled in `main.py` (lines 53-56)
- 📋 Ready to re-enable once MLflow PR #20784 is merged

**To Re-enable:** Uncomment lines 55-56 in `src/summit_sim/main.py`

---

## Quick Start

### Manual Evaluation (Working Now)

Run simulations, then evaluate offline in a notebook:

```python
from summit_sim.judges.rollup import compute_rollup_score

# After simulations complete and traces are logged
result = compute_rollup_score(session_id="your-session-id")
print(f"Overall score: {result.overall_score:.2f}")
print(f"Passed: {result.passed_criteria}/{result.total_criteria}")
```

### Automated Evaluation (Disabled)

Once MLflow fixes the bug:
1. Upgrade to MLflow 3.10.2+ (when PR #20784 is merged)
2. Uncomment `initialize_judges()` in `main.py`
3. Deploy

---

## Architecture

```
Student Action → Action Responder → DynamicTurnResult
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │   Manual/Offline    │
                              │  Judge Evaluation   │
                              │   (Working Now)     │
                              └─────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │   Rollup Compute    │
                              │   (Notebook/API)    │
                              └─────────────────────┘
```

---

## Evaluation Criteria

| Criterion | Judge | Weight | Description |
|-----------|-------|--------|-------------|
| score_in_range | Structure | 0.05 | Score between 0.0-1.0 |
| question_in_narrative_only | Structure | 0.10 | Narrative ends with question |
| feedback_no_harsh_language | Structure | 0.05 | Constructive tone |
| narrative_length | Structure | 0.05 | 3-5 sentences |
| score_milestone_justified | Scoring | 0.20 | Aligns with PAS rubric |
| score_not_over_awarded | Scoring | 0.15 | Reasonable score jumps |
| feedback_acknowledges_actions | Scoring | 0.05 | Mentions specific actions |
| was_correct_treatment_gate | Medical | 0.15 | Accurate treatment assessment |
| score_monotonic | Continuity | 0.10 | Score never decreases |
| narrative_reveals_progressively | Continuity | 0.10 | No repeated "new" facts |

**Total: 1.00**

---

## File Structure

```
src/summit_sim/
├── judges/
│   ├── __init__.py           # Public exports
│   ├── structure.py          # Structure judge (4 criteria)
│   ├── scoring.py            # Scoring judge (3 criteria)
│   ├── medical.py            # Medical judge (1 criterion)
│   ├── continuity.py         # Continuity judge (2 criteria)
│   ├── setup.py              # Registration (disabled)
│   ├── rollup.py             # Offline computation
│   └── utils.py              # Config + cache utilities
├── schemas.py                # RollupResult schema
└── main.py                   # Integration (disabled)
```

---

## Configuration

### Model Endpoint

Uses OpenRouter direct via LiteLLM:
```python
# src/summit_sim/judges/utils.py
JUDGE_MODEL_ENDPOINT = f"openrouter:/{settings.judge_model}"
```

Default model: `anthropic/claude-3-haiku` (configurable via `judge_model` setting)

### Judge Output Format

All judges return flat boolean dictionaries:
```python
# Structure judge returns:
{
    "score_in_range": True,
    "question_in_narrative_only": True,
    "feedback_no_harsh_language": False,
    "narrative_length": True
}
```

This satisfies MLflow's type requirements (`dict[str, bool]`).

---

## The MLflow Bug

**Issue:** Automated scoring fails when MLflow is configured with S3 artifact storage.

**Error:**
```
MlflowTracingException: Trace data not stored in tracking store
```

**Root Cause:** The automated scoring job (`OnlineTraceLoader`) only looks for traces in the SQL database. When S3 is configured, trace spans are stored in S3, causing the lookup to fail.

**Fix:** PR #20784 adds S3 fallback to the scoring job.

**Workarounds Attempted:**
1. ❌ `MLFLOW_TRACE_SPANS_LOCATION=TRACKING_STORE` - Didn't work
2. ✅ **Disable automated judges** - Current solution
3. ⏳ **Wait for MLflow fix** - Recommended long-term

---

## Usage

### Offline Evaluation (Current)

```python
from summit_sim.judges.rollup import (
    compute_rollup_score,
    compute_rollup_for_all_sessions
)

# Single session
result = compute_rollup_score("session-123")

# All sessions
results = compute_rollup_for_all_sessions()
```

### Single Turn Evaluation (For Prompt Testing)

```python
from summit_sim.judges.rollup import compute_judge_score_for_turn

# During prompt iteration
score_data = compute_judge_score_for_turn(turn_result)
print(f"Score: {score_data['overall_score']}")
```

---

## Re-enabling Automated Evaluation

Once MLflow PR #20784 is merged:

1. **Update MLflow:** Upgrade to version with the fix (likely 3.10.2+)
2. **Uncomment initialization:** In `src/summit_sim/main.py`:
   ```python
   # Change from:
   # DISABLED: MLflow bug #20782...
   # initialize_judges()
   
   # To:
   initialize_judges()
   logger.debug("Judges initialized")
   ```
3. **Deploy:** Automated evaluation will start working

**Automated Evaluation Behavior:**
- Trace judges run 1-2 minutes after each trace
- Session judge runs 5 minutes after session completion
- Filter: `tags.graph_type = 'simulation' AND tags.agent_name = 'action-responder'`

---

## Testing

All judge code is tested:
```bash
python -m pytest tests/ -v
# 102 tests pass
```

Key tests verify:
- Judge imports work correctly
- Rollup computation logic
- Weight validation (sums to 1.0)

---

## References

- **MLflow Bug:** [#20782](https://github.com/mlflow/mlflow/issues/20782)
- **Fix PR:** [#20784](https://github.com/mlflow/mlflow/pull/20784)
- **MLflow Docs:** https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/
- **Architecture:** `plans/high-level-arch.md`

---

## Changelog

### 2026-03-29 - Implementation Complete
- ✅ Implemented all 4 judges (structure, scoring, medical, continuity)
- ✅ Merged config.py into utils.py
- ✅ Changed to flat boolean output format (`dict[str, bool]`)
- ✅ Switched to OpenRouter direct endpoint
- ✅ Fixed `ScorerSamplingConfig` usage (filter_string inside config)
- ❌ **Disabled automated evaluation** due to MLflow bug #20782
- ✅ All 102 tests pass
- ✅ Ruff linting clean

### Known Issues
- Automated judges disabled until MLflow PR #20784 is merged
- Manual/offline evaluation works perfectly
