# Story 1.1: PydanticAI Agent & MLflow Tracing

## Overview

Implement the Simulation Agent with PydanticAI, define strict Pydantic schemas, and configure MLflow tracing to the homelab server.

## TDD Verification Goals

1. Write a Python script passing a hardcoded ScenarioDraft and a student action ("apply a tourniquet")
2. Assert it returns a valid SimulationTurn object
3. MLflow check: Open homelab MLflow UI and assert a trace appears showing the prompt, structured output, latency, and token usage

## Implementation Steps

### 1. Dependencies

Add to `pyproject.toml` dependencies:

```toml
dependencies = [
    "pydantic-ai>=0.0.50",
    "mlflow>=2.20.0",
    "pydantic-settings>=2.0.0",
]
```

### 2. Settings Configuration

**File**: `src/summit_sim/settings.py`

Create Pydantic Settings with:
- MLflow tracking URI (env: `MLFLOW_TRACKING_URI`, default: `http://localhost:5000`)
- OpenRouter API key (env: `OPENROUTER_API_KEY`)
- OpenRouter base URL (default: `https://openrouter.ai/api/v1`)
- Default model configuration

### 3. Pydantic Schemas

**File**: `src/summit_sim/schemas.py`

Define schemas based on high-level architecture:

**ScenarioDraft**:
- title: str
- setting: str
- patient_summary: str
- hidden_truth: str
- starting_conditions: str
- learning_objectives: list[str]

**SimulationTurn**:
- narrative_text: str
- feedback_on_last_action: str
- updated_hidden_state: dict[str, Any]
- updated_scene_state: dict[str, Any]
- available_actions: list[str]
- learning_moments: list[str]
- is_complete: bool

### 4. Simulation Agent

**File**: `src/summit_sim/agents/simulation.py`

Implementation:
- PydanticAI agent using OpenRouter
- Hardcoded model config (as per discussion - flexibility comes later)
- Function signature: `async def simulate_turn(scenario: ScenarioDraft, student_action: str, transcript: list[str] | None = None) -> SimulationTurn`
- Enable MLflow GenAI autologging
- System prompt describing the simulation behavior

### 5. Test Script

**File**: `tests/test_simulation_agent.py`

Test implementation:
- Create hardcoded ScenarioDraft instance
- Call `simulate_turn()` with action "apply a tourniquet"
- Assert return type is SimulationTurn
- Assert required fields are populated
- Print instructions for manual MLflow UI verification

Example test:
```python
@pytest.mark.asyncio
async def test_simulation_agent_returns_valid_turn():
    scenario = ScenarioDraft(
        title="Hiking Accident",
        setting="Rocky Mountain trail, 8,000ft elevation",
        patient_summary="42yo male with severe leg laceration",
        hidden_truth="Arterial bleed requiring immediate tourniquet",
        starting_conditions="Patient conscious, bleeding profusely",
        learning_objectives=["Control severe bleeding", "Assess consciousness"]
    )
    
    result = await simulate_turn(scenario, "apply a tourniquet")
    
    assert isinstance(result, SimulationTurn)
    assert result.narrative_text
    assert result.is_complete is not None
```

### 6. Manual MLflow Verification Checklist

After running tests:

1. Open MLflow UI at your homelab server URL
2. Navigate to "Experiments" or "Traces"
3. Verify trace appears for test run
4. Check trace contains:
   - Prompt text (system + user messages)
   - Structured output (SimulationTurn JSON)
   - Latency metrics
   - Token usage (input/output counts)
5. Document any connection issues or configuration needed

## Configuration

Environment variables (create `.env` file):

```bash
# MLflow
MLFLOW_TRACKING_URI=http://your-homelab-server:5000

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
```

## Notes

- Keep prompts simple initially - focus on schema validation
- Model assignment is hardcoded for now, config file comes later
- No complex validation rules on schemas yet
- Manual MLflow verification is acceptable for this story
- Schema file is named `schemas.py` (preferred over models.py to avoid confusion with ML models)

## Success Criteria

- [ ] Dependencies added and installed
- [ ] Settings module loads from environment
- [ ] Schemas defined with proper types
- [ ] Simulation Agent returns SimulationTurn
- [ ] Test passes with hardcoded data
- [ ] MLflow trace visible in UI with required fields
- [ ] All code passes ruff linting
- [ ] Test coverage ≥80%
