# Story 1.2: LangGraph Simulation Workflow

## Overview

Implement a LangGraph workflow that orchestrates the simulation loop with human-in-the-loop interrupts for student choices. The graph manages state, calls the Simulation Feedback Agent, and advances through turns until completion.

**Architecture Decision**: Using LangGraph's native `interrupt()` for human-in-the-loop control, Pydantic models for state (consistent with project), and integer-based turn IDs for simplicity.

---

## Implementation Status

**Status**: PENDING
**Test Coverage**: TBD
**All Tests**: TBD

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

```python
class AppState(BaseModel):
    """LangGraph state for simulation workflow."""
    
    scenario_draft: ScenarioDraft
    current_turn_id: int
    transcript: list[TranscriptEntry]
    is_complete: bool = False
    key_learning_moments: list[str] = Field(default_factory=list)

class TranscriptEntry(BaseModel):
    """Single entry in simulation transcript."""
    
    turn_id: int
    turn_narrative: str
    choice_id: str
    choice_description: str
    feedback: str
    learning_moments: list[str]
    next_turn_id: int | None
```

---

### Phase 4: Graph Implementation

**File**: `src/summit_sim/graphs/simulation.py`

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
   - Initialize graph: `graph = create_simulation_graph()`
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

- [ ] LangGraph added via `uv add`
- [ ] Schemas updated to use integer turn IDs
- [ ] `AppState` Pydantic model defined with full transcript context
- [ ] Simulation graph implemented with 5 nodes
- [ ] Graph uses `interrupt()` for human-in-the-loop
- [ ] Unit tests pass (full cycle test with mocked agent)
- [ ] Notebook demonstrates E2E flow with real agents
- [ ] MLflow traces show multi-step execution under Parent Run
- [ ] Error handling for invalid turn references
- [ ] All code passes ruff linting
- [ ] Coverage ≥80%

---

## Files to Create/Modify

### New Files
1. `src/summit_sim/graphs/__init__.py` - Package init
2. `src/summit_sim/graphs/state.py` - AppState and TranscriptEntry
3. `src/summit_sim/graphs/simulation.py` - LangGraph implementation
4. `tests/test_simulation_graph.py` - Graph unit tests
5. `notebooks/story-1-2-simulation-graph.ipynb` - E2E integration test

### Modified Files
1. `src/summit_sim/schemas.py` - Update to integer turn IDs
2. `pyproject.toml` - Add langgraph dependency (via uv)

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
- **State Type**: Using Pydantic BaseModel instead of TypedDict (consistent with project)
- **Interrupt Pattern**: Native LangGraph `interrupt()` with `Command(resume=...)`
- **Transcript**: Full context captured for debrief analysis
- **Testing**: Simple full-cycle test preferred over per-node tests
- **MLflow**: Manual verification in notebook (not automated test)

---

## Next Steps

1. **Run `uv add langgraph`** to install dependency
2. **Update schemas** to use integer turn IDs
3. **Implement AppState** Pydantic model
4. **Build simulation graph** with interrupt handling
5. **Write unit tests** with mocked agent
6. **Create E2E notebook** with real agents
7. **Verify MLflow** traces show multi-step execution

---

## Dependencies

```toml
[project]
dependencies = [
    "langgraph>=0.2.0",  # Will be added via uv
    "mlflow>=3.10.1",
    "pydantic-ai>=1.70.0",
    "pydantic-settings>=2.13.1",
    "pytest-asyncio>=1.3.0",
]
```

---

## Related Documents

- `high-level-arch.md` - Overall system architecture
- `story-1-1-implementation.md` - Previous story (PydanticAI agents)
- `agent-configuration-patterns.md` - Shared agent patterns
