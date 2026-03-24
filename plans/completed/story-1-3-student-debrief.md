# Story 1.3: Student Simulation Debrief Generation & Metrics

## Overview

Implement the Debrief Agent that analyzes completed student simulation runs. The agent takes the final `SimulationState` (formerly AppState), generates a structured `DebriefReport` with performance analysis, and logs metrics to MLflow.

**Scope**: Student simulation debrief only (not generator debrief).

---

## Implementation Status

**Status**: NOT STARTED
**Estimated Complexity**: Medium
**Dependencies**: Story 1.2 (Simulation Graph)

---

## Architecture

### Debrief Flow

```
Simulation Complete
       ↓
   Debrief Node (new)
       ↓
   DebriefAgent.analyze()
       ↓
   DebriefReport
       ↓
   ┌─────────────┬─────────────┐
   ↓             ↓             ↓
MLflow       Return to      Store in
Metrics      Caller         State
```

### Key Design Decisions

1. **Separate debrief node**: Runs after simulation graph completes (check_completion → __end__)
2. **State rename**: `AppState` → `SimulationState` (more accurate)
3. **ID structure**: 
   - `scenario_id`: Unique per scenario run (primary link)
   - `class_id`: Optional grouping (e.g., for classes with multiple scenarios)
4. **Metrics**: Separate MLflow run (simulation phase) with `scenario_id` + optional `class_id` tags
5. **Pattern consistency**: DebriefAgent follows same `get_agent()` pattern as generator/simulation

---

## Pre-Requisites

These changes must be completed **before** implementing the DebriefAgent. They establish the foundational structures and ID system needed for proper traceability.

### Pre-Req 1: State Rename

**File**: `src/summit_sim/graphs/state.py`

Rename `AppState` → `SimulationState` for accuracy:
- Current name suggests it represents the entire application
- Actually only represents simulation phase state
- More descriptive and reduces confusion

**Changes**:
1. Rename TypedDict class in `state.py`
2. Update all imports across codebase
3. Update test files
4. Keep `AppState` as deprecated alias for backward compatibility (optional)

### Pre-Req 2: ID Structure Implementation

**Rationale**: Need to distinguish between:
- **scenario_id**: Unique identifier per scenario run (links generation → simulation → debrief)
- **class_id**: Optional grouping tag (e.g., "Wilderness Class 2024" can have multiple scenarios)

**This enables**: One class doing multiple scenarios, or individual scenarios without class grouping.

#### Changes to `HostConfig` (`src/summit_sim/schemas.py`):
```python
class HostConfig(BaseModel):
    # ... existing fields ...
    
    class_id: str | None = Field(
        default=None,
        description="Optional class grouping ID (e.g., 'class-2024-wfa')"
    )
    
    # scenario_id will be auto-generated during scenario creation
```

#### Changes to `SimulationState` (`src/summit_sim/graphs/state.py`):
```python
class SimulationState(TypedDict):
    # ... existing fields ...
    
    scenario_id: str  # Required - unique per scenario run
    class_id: str | None  # Optional - for grouping multiple scenarios
```

**Note**: `scenario_id` should be generated during scenario initialization (in generation phase), not in HostConfig, since one host config could generate multiple scenarios across different runs.

### Pre-Req 3: TranscriptEntry Enhancement

**File**: `src/summit_sim/graphs/state.py`

Add `was_correct` field to track choice correctness:
```python
class TranscriptEntry(TypedDict):
    # ... existing fields ...
    
    was_correct: bool  # Whether the chosen option was the medically optimal choice
```

**Rationale**: Enables deterministic score calculation without re-querying the scenario structure. The simulation graph already knows if a choice was correct (from `ChoiceOption.is_correct`), so we should capture this in the transcript.

### Pre-Req 4: Scenario ID Generation

**File**: `src/summit_sim/agents/generator.py` or `src/summit_sim/graphs/simulation.py`

Generate unique `scenario_id` when scenario is created:
```python
def generate_scenario_id() -> str:
    """Generate unique scenario identifier."""
    return f"scn-{uuid.uuid4().hex[:8]}"  # e.g., "scn-a3f8d2e9"
```

**Integration**:
- Generation phase: Create scenario_id and pass to SimulationState
- Simulation phase: Use scenario_id for MLflow tagging
- Debrief phase: Use scenario_id for report correlation

### Pre-Req 5: MLflow Session Update

**File**: `src/summit_sim/tracing.py`

Update `simulation_session` context manager:
```python
@contextmanager
def simulation_session(
    config: HostConfig,
    scenario_id: str,  # NEW: Required parameter
    session_id: str | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, None]:
    """Context manager for simulation session with MLflow parent run."""
    session_id = session_id or str(uuid.uuid4())
    session_name = generate_session_name(config, phase="sim")
    
    with mlflow.start_run(run_name=session_name):
        # Log session-level parameters
        mlflow.log_params({
            "session_id": session_id,
            "scenario_id": scenario_id,  # NEW
            "class_id": config.class_id,  # NEW (may be None)
            # ... existing params ...
        })
        
        # Set tags for easy filtering
        mlflow.set_tags({
            "session_type": "simulation",
            "scenario_id": scenario_id,  # NEW
            # ... conditionally add class_id if present
        })
        
        if config.class_id:
            mlflow.set_tag("class_id", config.class_id)
```

### Pre-Req 6: Update Simulation Graph

**File**: `src/summit_sim/graphs/simulation.py`

Update graph to accept and propagate scenario_id:
1. Add scenario_id to initial state creation
2. Ensure it's passed through graph state
3. Update all state references from AppState to SimulationState

### Pre-Req Checklist

Before starting Story 1.3 implementation:

- [ ] Rename `AppState` → `SimulationState`
- [ ] Add `was_correct` field to `TranscriptEntry`
- [ ] Add `class_id` (optional) to `HostConfig`
- [ ] Add `scenario_id` and `class_id` to `SimulationState`
- [ ] Implement `scenario_id` generation in generation phase
- [ ] Update `simulation_session` to log both IDs
- [ ] Update all imports and test files
- [ ] Run full test suite to ensure no regressions
- [ ] Verify MLflow tagging works with new IDs

---

## Implementation Tasks

### Phase 1: Schema Updates

**File**: `src/summit_sim/schemas.py`

#### 1.1 Rename AppState references
- Rename `AppState` to `SimulationState` in `src/summit_sim/graphs/state.py`
- Update all imports across codebase
- Update tests

#### 1.2 Add DebriefReport schema
```python
class DebriefReport(BaseModel):
    """Structured debrief report analyzing student simulation performance."""
    
    summary: str = Field(..., description="Executive summary of the simulation run")
    key_mistakes: list[str] = Field(..., description="Critical errors made during the simulation")
    strong_actions: list[str] = Field(..., description="Decisions the student handled well")
    best_next_actions: list[str] = Field(..., description="Recommendations for future scenarios")
    teaching_points: list[str] = Field(..., description="Key learning concepts to reinforce")
    completion_status: Literal["pass", "fail"] = Field(
        ..., description="Overall pass/fail based on performance"
    )
    final_score: float = Field(
        ..., ge=0, le=100, description="Percentage score (correct choices / total turns * 100)"
    )
```

#### 1.3 Add scenario_id to schemas
Update these schemas to include IDs:

**HostConfig**:
- Keep `class_id` (optional, for grouping)
- Add `scenario_id` (auto-generated, unique per run)

**SimulationState**:
- Add `scenario_id` field
- Add `class_id` field (optional)

### Phase 2: State Rename

**Files to update**:
1. `src/summit_sim/graphs/state.py` - Rename `AppState` → `SimulationState`
2. `src/summit_sim/graphs/simulation.py` - Update all references
3. `src/summit_sim/graphs/__init__.py` - Export new name
4. All test files that import AppState

### Phase 3: Debrief Agent

**File**: `src/summit_sim/agents/debrief.py`

#### 3.1 Implementation
```python
DEBRIEF_SYSTEM_PROMPT = """You are an expert wilderness first aid educator analyzing a student's simulation performance.

Your task is to review the complete simulation transcript and provide a constructive, learning-focused debrief.

Analysis guidelines:
1. Review each turn's choice and the AI feedback provided
2. Identify patterns in decision-making
3. Tally correct vs incorrect choices for scoring
4. Highlight both strengths and areas for improvement
5. Provide specific, actionable recommendations

Scoring:
- Calculate final_score as: (number of correct choices / total turns) * 100
- Determine completion_status: "pass" if final_score >= 70, "fail" otherwise

Tone: Encouraging but honest. Focus on learning, not grading."""


async def generate_debrief(
    transcript: list[TranscriptEntry],
    scenario_draft: ScenarioDraft,
    scenario_id: str,
) -> DebriefReport:
    """Generate debrief report from completed simulation.
    
    Args:
        transcript: Complete simulation transcript
        scenario_draft: Original scenario for context
        scenario_id: Unique scenario identifier
        
    Returns:
        Structured debrief report with score and analysis
    """
    agent = get_agent(
        agent_name="debrief",
        output_type=DebriefReport,
        system_prompt=DEBRIEF_SYSTEM_PROMPT,
        reasoning_effort="medium",
    )
    
    # Build analysis prompt
    prompt = _build_debrief_prompt(transcript, scenario_draft)
    
    result = await agent.run(prompt)
    return result.output


def _build_debrief_prompt(
    transcript: list[TranscriptEntry],
    scenario_draft: ScenarioDraft,
) -> str:
    """Build prompt for debrief agent."""
    # Include scenario context
    # Include full transcript with choices and feedback
    # Include correct/incorrect tallies for verification
    pass
```

#### 3.2 Calculate score deterministically
```python
def calculate_score(transcript: list[TranscriptEntry]) -> float:
    """Calculate percentage score from transcript.
    
    Score = (correct_choices / total_turns) * 100
    """
    if not transcript:
        return 0.0
    
    total_turns = len(transcript)
    # Need to track if choice was correct - may need to add to TranscriptEntry
    correct_choices = sum(1 for entry in transcript if entry.get("was_correct", False))
    
    return (correct_choices / total_turns) * 100
```

**Note**: May need to add `was_correct: bool` to `TranscriptEntry` to track correctness.

### Phase 4: Graph Integration

**File**: `src/summit_sim/graphs/simulation.py`

#### 4.1 Add debrief node
```python
async def generate_debrief_node(state: SimulationState) -> dict:
    """Generate debrief report after simulation completes."""
    debrief_report = await generate_debrief(
        transcript=state["transcript"],
        scenario_draft=state["scenario_draft"],
        scenario_id=state["scenario_id"],
    )
    
    return {"debrief_report": debrief_report}
```

#### 4.2 Update graph flow
Add debrief node that runs after completion:
```python
workflow.add_node("generate_debrief", generate_debrief_node)
workflow.add_edge("update_state", "generate_debrief")
workflow.add_edge("generate_debrief", "check_completion")
```

Or alternatively, debrief could be a separate graph/node called after the main simulation graph completes.

**Decision**: For cleaner separation, the debrief should probably be a separate callable (not in the cyclic graph) that's invoked after the simulation completes.

### Phase 5: MLflow Integration

**File**: `src/summit_sim/tracing.py`

#### 5.1 Add debrief logging function
```python
def log_debrief_metrics(
    debrief_report: DebriefReport,
    scenario_id: str,
    class_id: str | None = None,
) -> None:
    """Log debrief metrics to current MLflow run.
    
    Should be called within simulation_session context.
    """
    mlflow.log_metrics({
        "final_score": debrief_report.final_score,
        "is_complete": 1.0,  # Debrief was generated
    })
    
    mlflow.set_tags({
        "pass_fail": debrief_report.completion_status,
        "scenario_id": scenario_id,
    })
    
    if class_id:
        mlflow.set_tag("class_id", class_id)
```

#### 5.2 Update simulation_session
- Add `scenario_id` parameter
- Generate scenario_id if not provided
- Log scenario_id in params and tags

### Phase 6: Testing

**File**: `tests/test_debrief.py`

#### 6.1 Unit tests
```python
class TestDebriefAgent:
    """Test debrief agent functionality."""
    
    async def test_generate_debrief_returns_report(self):
        """Test agent returns DebriefReport."""
        # Arrange: Mock completed state
        # Act: Call generate_debrief
        # Assert: Returns DebriefReport
        
    async def test_debrief_fail_status(self):
        """Test report shows fail when mistakes present."""
        # Arrange: Mock state with mostly incorrect choices
        # Act: Generate debrief
        # Assert: completion_status == "fail"
        
    async def test_final_score_calculation(self):
        """Test score is percentage of correct choices."""
        # Arrange: 3 turns, 1 correct
        # Act: Generate debrief
        # Assert: final_score == 33.33 (or similar)
        
    async def test_debrief_includes_mistakes(self):
        """Test report identifies specific mistakes."""
        # Arrange: State with known incorrect choices
        # Act: Generate debrief
        # Assert: key_mistakes contains expected mistakes


class TestMLflowLogging:
    """Test MLflow metrics logging."""
    
    def test_final_score_logged_as_metric(self):
        """Test final_score is logged as MLflow metric."""
        
    def test_pass_fail_logged_as_tag(self):
        """Test pass_fail is logged as MLflow tag."""
        
    def test_scenario_id_logged(self):
        """Test scenario_id is logged in tags."""
```

#### 6.2 Integration test (optional)
Create notebook `notebooks/story-1-3-debrief.ipynb`:
1. Run full simulation
2. Generate debrief
3. Verify MLflow metrics
4. Display DebriefReport

---

## Success Criteria

- [ ] `DebriefReport` schema defined with all fields
- [ ] `AppState` renamed to `SimulationState`
- [ ] `scenario_id` added to HostConfig and SimulationState
- [ ] `class_id` kept as optional grouping field
- [ ] DebriefAgent implemented following `get_agent()` pattern
- [ ] Score calculation: `(correct_choices / total_turns) * 100`
- [ ] MLflow logs `final_score` as metric
- [ ] MLflow logs `pass_fail` as tag
- [ ] MLflow logs `scenario_id` for trace linking
- [ ] Unit tests with mocked agent (TDD approach)
- [ ] Tests verify fail status with known mistakes
- [ ] Tests verify MLflow metric logging
- [ ] All tests pass (≥80% coverage)
- [ ] Ruff linting passes

---

## Files to Create/Modify

### New Files
1. `src/summit_sim/agents/debrief.py` - DebriefAgent implementation
2. `tests/test_debrief.py` - Debrief agent tests
3. `notebooks/story-1-3-debrief.ipynb` - Integration notebook
4. `plans/backlog/story-1-4-generator-debrief.md` - Generator debrief backlog

### Modified Files
1. `src/summit_sim/schemas.py` - Add DebriefReport, update IDs
2. `src/summit_sim/graphs/state.py` - Rename AppState → SimulationState, add IDs
3. `src/summit_sim/graphs/simulation.py` - Update state references
4. `src/summit_sim/graphs/__init__.py` - Export SimulationState
5. `src/summit_sim/tracing.py` - Add debrief logging, scenario_id support
6. `tests/test_simulation_graph.py` - Update state references
7. `tests/test_schemas.py` - Add DebriefReport tests

---

## TDD Verification Plan

### Test 1: Debrief Returns Fail Status
```python
async def test_debrief_returns_fail_with_mistakes():
    """Pass mock completed state with known mistakes, assert 'fail' status."""
    # Create transcript with 3 turns, 1 correct choice
    transcript = create_mock_transcript(correct_count=1, total_turns=3)
    
    mock_report = DebriefReport(
        summary="Test summary",
        key_mistakes=["Mistake 1", "Mistake 2"],
        strong_actions=["Good action"],
        best_next_actions=["Do this next"],
        teaching_points=["Learn this"],
        completion_status="fail",
        final_score=33.33,
    )
    
    with patch("summit_sim.agents.debrief.get_agent") as mock_get_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = Mock(output=mock_report)
        mock_get_agent.return_value = mock_agent
        
        result = await generate_debrief(transcript, mock_scenario, "scenario-123")
        
    assert result.completion_status == "fail"
    assert result.final_score < 70
```

### Test 2: MLflow Metrics
```python
def test_mlflow_logs_final_score():
    """Verify run logs final_score as metric."""
    with mlflow.start_run():
        log_debrief_metrics(
            DebriefReport(final_score=80.0, completion_status="pass", ...),
            scenario_id="test-123",
        )
        
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        assert "final_score" in run.data.metrics
        assert run.data.metrics["final_score"] == 80.0

def test_mlflow_logs_pass_fail_tag():
    """Verify pass_fail logged as tag."""
    with mlflow.start_run():
        log_debrief_metrics(
            DebriefReport(completion_status="fail", ...),
            scenario_id="test-123",
        )
        
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        assert run.data.tags.get("pass_fail") == "fail"
```

---

## Notes

- **Breaking change**: Renaming AppState will require updates to all existing imports
- **TranscriptEntry**: May need `was_correct` field for deterministic score calculation
- **Debrief timing**: Debrief runs once after simulation completes (not part of cyclic graph)
- **Score threshold**: Using 70% as pass threshold (configurable in future)
- **MLflow integration**: Extends existing tracing.py patterns
- **Generator debrief**: Intentionally out of scope (see backlog file)

---

## Next Steps

1. Review and approve plan
2. Implement schema changes (DebriefReport, ID fields)
3. Rename AppState → SimulationState
4. Implement DebriefAgent
5. Add MLflow logging
6. Write tests (TDD approach)
7. Create integration notebook
8. Update documentation

---

## Dependencies

- Story 1.1 (PydanticAI Agents) - Complete
- Story 1.2 (Simulation Graph) - Complete
- MLflow tracing configuration - Complete
