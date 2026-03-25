# Story 1.2: LangGraph Simulation Workflow

## Overview

Implement a LangGraph workflow that orchestrates the simulation loop with human-in-the-loop interrupts for student choices. The graph manages state, calls the Simulation Feedback Agent, and advances through turns until completion.

**Architecture Decision**: Using LangGraph's native `interrupt()` for human-in-the-loop control, Pydantic models for state (consistent with project), and integer-based turn IDs for simplicity.

---

## Implementation Status

**Status**: COMPLETED ✅
**Test Coverage**: 12 unit tests + 34 total tests passing
**All Tests**: ✅ PASSING

---

## Architecture

### Simulation Graph Flow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ initialize_state│────▶│ present_turn │────▶│  interrupt  │
└─────────────────┘     └──────────────┘     └─────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ check_completion│◀────│ update_state │◀────│ process_turn│
└────────┬────────┘     └──────────────┘     └─────────────┘
         │
    ┌────┴────┐
    │  END    │ (if complete)
    └─────────┘
```

**Key Design Decisions**:
1. **Pydantic AppState** - Consistent with project patterns, better validation
2. **Integer Turn IDs** - Simple 0, 1, 2 indexing instead of strings
3. **Native interrupt()** - Built-in LangGraph feature for human-in-the-loop
4. **Full transcript context** - Complete turn info for debrief analysis

---

## Implementation Tasks

### Phase 1: Dependencies

**Command**:
```bash
uv add langgraph
```

---

### Phase 2: Schema Updates

**File**: `src/summit_sim/schemas.py`

Update type signatures:
- `turn_id: str` → `turn_id: int` (in `ScenarioTurn`)
- `next_turn_id: str | None` → `next_turn_id: int | None` (in `ChoiceOption`)
- `starting_turn_id: str` → `starting_turn_id: int` (in `ScenarioDraft`)
- Update `get_turn()` method signature to accept `int`

---

### Phase 3: AppState Definition

**File**: `src/summit_sim/graphs/state.py`

**Implementation Note**: Used TypedDict instead of Pydantic BaseModel for better LangGraph compatibility.

```python
class TranscriptEntry(TypedDict):
    """Single entry in simulation transcript."""
    
    turn_id: int
    turn_narrative: str
    choice_id: str
    choice_description: str
    feedback: str
    learning_moments: list[str]
    next_turn_id: int | None

class AppState(TypedDict):
    """LangGraph state for simulation workflow."""
    
    scenario_draft: ScenarioDraft
    current_turn_id: int
    transcript: Annotated[list[TranscriptEntry], _add]
    is_complete: bool
    key_learning_moments: Annotated[list[str], _add]
    last_selected_choice: Any
    simulation_result: Any
    class_id: str  # Added for MLflow trace linking
```

---

### Phase 4: Graph Implementation

**File**: `src/summit_sim/graphs/student.py`

#### Nodes

1. **`initialize_state(state: AppState)`**
   - Validates `starting_turn_id` exists in scenario
   - Sets `current_turn_id = scenario.starting_turn_id`
   - Initializes empty transcript

2. **`present_turn(state: AppState)`**
   - Gets current turn from scenario
   - Displays narrative and choices
   - Calls `interrupt()` with choice options
   - Returns updated state with `last_student_choice_id`

3. **`process_turn(state: AppState)`**
   - Calls `process_choice()` Simulation Agent
   - Passes: scenario, current_turn, selected_choice
   - Returns: `SimulationResult` with feedback

4. **`update_state(state: AppState)`**
   - Creates `TranscriptEntry` with full context
   - Appends to `state.transcript`
   - Extends `key_learning_moments`
   - Updates `current_turn_id` from `next_turn_id`

5. **`check_completion(state: AppState)`**
   - Returns `"__end__"` if `next_turn_id is None`
   - Returns `"present_turn"` otherwise

#### Error Handling
- `ValueError` if `current_turn_id` not found in scenario
- `ValueError` if choice's `next_turn_id` doesn't exist

---

### Phase 5: Testing

**File**: `tests/test_simulation_graph.py`

Simple unit test approach:

```python
async def test_full_simulation_cycle():
    """Test complete 3-turn simulation with mocked agent."""
    # Arrange
    scenario = create_mock_scenario(turn_ids=[0, 1, 2])
    mock_process_choice = create_mock_responses([
        {"next_turn_id": 1, "is_complete": False},
        {"next_turn_id": 2, "is_complete": False},
        {"next_turn_id": None, "is_complete": True},
    ])
    
    # Act
    result = await run_simulation_graph(scenario, mock_process_choice)
    
    # Assert
    assert len(result.transcript) == 3
    assert result.transcript[0].turn_id == 0
    assert result.transcript[2].next_turn_id is None
    assert result.is_complete is True
```

**Test Strategy**:
- One main test for full cycle
- Mock `process_choice()` agent calls
- Verify transcript structure
- Verify graph termination

---

### Phase 6: E2E Integration Notebook

**File**: `notebooks/story-1-2-simulation-graph.ipynb`

**Sections**:

1. **Setup**
   - Import graph and state
   - Initialize MLflow autologging

2. **Generate Scenario**
   - Call `generate_scenario()` with real agent
   - Display generated turns

3. **Run Simulation**
   - Initialize graph: `graph = create_student_graph()`
   - Configure checkpointing (if needed)
   - Start with: `state = await graph.ainvoke(initial_state)`
   - Handle interrupts:
   ```python
   for event in graph.astream(state):
       if "__interrupt__" in event:
           # Display current turn and choices
           choice_id = input("Select choice (1-3): ")
           state = await graph.ainvoke(
               Command(resume={"choice_id": choice_id})
           )
   ```

4. **Verify MLflow**
   - Show single Parent Run with child spans
   - Display multi-step execution trace

5. **Display Results**
   - Show complete transcript
   - List accumulated learning moments

---

## Success Criteria

- [x] LangGraph added via `uv add` (also added `langchain` for MLflow tracing)
- [x] Schemas updated to use integer turn IDs
- [x] `AppState` TypedDict defined with full transcript context (TypedDict chosen over Pydantic for LangGraph compatibility)
- [x] Simulation graph implemented with 5 nodes
- [x] Graph uses `interrupt()` for human-in-the-loop
- [x] Unit tests pass (12 tests, full cycle with mocked agent)
- [x] Notebook demonstrates E2E flow with real agents
- [x] MLflow traces show multi-step execution under Parent Run
- [x] Error handling for invalid turn references
- [x] All code passes ruff linting
- [x] Coverage ≥80%

### Additional Features Implemented (Not in Original Plan)

- [x] **class_id linking**: Added `class_id` field to link generation and simulation traces in MLflow
- [x] **Unified tracing module**: Created `src/summit_sim/tracing.py` for centralized MLflow configuration
- [x] **Session management**: Context managers for parent runs with descriptive naming (`sim-{class_id}-{activity}-{participants}p-{difficulty}`)
- [x] **Cross-phase trace linking**: Filter by `tags.class_id` in MLflow UI to see complete flow

---

## Files Created/Modified

### New Files
1. ✅ `src/summit_sim/graphs/__init__.py` - Package init
2. ✅ `src/summit_sim/graphs/state.py` - AppState and TranscriptEntry (TypedDict)
3. ✅ `src/summit_sim/graphs/student.py` - LangGraph implementation with 5 nodes
4. ✅ `src/summit_sim/tracing.py` - MLflow tracing utilities with session management
5. ✅ `tests/test_simulation_graph.py` - Graph unit tests (12 tests)
6. ✅ `notebooks/story-1-2-simulation-graph.ipynb` - E2E integration test

### Modified Files
1. ✅ `src/summit_sim/schemas.py` - Update to integer turn IDs, added `class_id` to HostConfig
2. ✅ `pyproject.toml` - Added `langgraph`, `langchain`, and notebook-specific ruff ignores
3. ✅ `plans/high-level-arch.md` - Replaced `room_id` with `class_id`

---

## Testing Approach

### Unit Tests (Mocked)
- Create `ScenarioDraft` fixture with 3 turns (IDs: 0, 1, 2)
- Mock `process_choice` to return predictable `SimulationResult`
- Run graph programmatically through 3 turns
- Assert transcript updates and graph termination

### Integration Test (Notebook)
- Use real `generate_scenario()` agent
- Manual interaction via notebook cells
- Verify MLflow traces in UI
- Full scenario walkthrough

---

## Notes

- **Turn IDs**: Changed from strings to integers (0, 1, 2) for simplicity
- **State Type**: Using TypedDict instead of Pydantic BaseModel (better LangGraph compatibility)
- **Interrupt Pattern**: Native LangGraph `interrupt()` with `Command(resume=...)`
- **Transcript**: Full context captured for debrief analysis
- **Testing**: Simple full-cycle test preferred over per-node tests
- **MLflow**: 
  - Manual verification in notebook (not automated test)
  - Uses `mlflow.langchain.autolog()` for LangGraph tracing
  - Context propagation warnings expected with async + interrupt pattern
- **class_id**: Added for cross-phase trace linking (generation → simulation)
- **Session Names**: Format `sim-{class_id}-{activity}-{participants}p-{difficulty}`

## Known Issues / Limitations

- **MLflow Context Warnings**: Async context propagation issues with `interrupt()` - traces still functional
- **Trace Flattening**: Some traces may not nest perfectly due to async context switching
- **Interrupt Exceptions**: Expected behavior - not actual errors

---

## Completed Work Summary

All implementation tasks completed successfully:

1. ✅ Installed langgraph and langchain dependencies
2. ✅ Updated schemas to use integer turn IDs (0, 1, 2)
3. ✅ Implemented AppState as TypedDict with transcript support
4. ✅ Built simulation graph with 5 nodes and interrupt handling
5. ✅ Created comprehensive unit tests (12 tests, all passing)
6. ✅ Created E2E notebook demonstrating full flow
7. ✅ Implemented MLflow tracing with class_id linking

## Ready for Next Story

This story is complete. Next steps per high-level-arch.md:
- Validation judges (Safety, Realism, Pedagogy)
- Host review workflow
- Student multi-user support
- Debrief generation

---

## Dependencies

```toml
[project]
dependencies = [
    "langgraph>=1.1.3",    # Added via uv
    "langchain>=1.2.13",   # Required for MLflow LangGraph tracing
    "mlflow>=3.10.1",
    "pydantic-ai>=1.70.0",
    "pydantic-settings>=2.13.1",
    "pytest-asyncio>=1.3.0",
]
```

**Note**: `langchain` added for MLflow integration (`mlflow.langchain.autolog()`) since LangGraph traces through LangChain.

---

## Related Documents

- `high-level-arch.md` - Overall system architecture
- `story-1-1-implementation.md` - Previous story (PydanticAI agents)
- `agent-configuration-patterns.md` - Shared agent patterns
