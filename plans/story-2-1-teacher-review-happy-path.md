# Story 2.1: Teacher Review Graph - Happy Path

**Epic**: Epic 2: Web POC with Teacher Review  
**Status**: Ready for Implementation  
**Estimated Effort**: 1-2 days  

## Overview

Implement the teacher review workflow with human-in-the-loop (HITL) via LangGraph's `interrupt()` pattern. This is the foundation of the teacher persona flow - teachers generate scenarios, review them, approve immediately (happy path only), and receive a shareable link for students.

**Scope**: Happy path only - generate → review → approve → share. No decline/regenerate logic (deferred to Story 2.3).

---

## Success Criteria

- [ ] `TeacherReviewState` defined with full schema (including future-use fields)
- [ ] Teacher graph executes: initialize → generate → interrupt → approve → END
- [ ] Notebook demonstrates full teacher flow end-to-end
- [ ] Chainlit renders config form and review screen
- [ ] Unique URL generated with scenario_id and class_id
- [ ] MLflow logs generation with `sme_approved: true` tag
- [ ] Unit and integration tests pass (≥80% coverage)
- [ ] Ruff linting passes

---

## Architecture

### Graph Flow

```
┌─────────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│initialize_teacher_session│───▶│generate_scenario │───▶│present_for_review│
│        (node)           │    │     (node)       │    │    (node)       │
└─────────────────────────┘    └──────────────────┘    └────────┬────────┘
                                                                │
                                                     [interrupt()]
                                                                │
                                          ┌─────────────────────┘
                                          │ Command(resume=
                                          │   {"decision": "approve"})
                                          ▼
                                       [END]
```

### State Definition

Add to `src/summit_sim/graphs/state.py`:

```python
class TeacherReviewState(TypedDict):
    """LangGraph state for teacher review workflow."""
    teacher_config: TeacherConfig
    scenario_draft: ScenarioDraft | None
    scenario_id: str
    class_id: str
    retry_count: int  # Always 0 in happy path, used in 2.3
    feedback_history: Annotated[list[str], append_reducer]  # Empty in happy path
    approval_status: str | None  # "approved" after resume
```

### Nodes

**1. initialize_teacher_session**
- Input: `teacher_config: TeacherConfig`
- Actions:
  - Generate `scenario_id` using `generate_scenario_id()`
  - Generate `class_id` using `generate_class_id()`
  - Initialize state with `retry_count=0`, empty `feedback_history`
- Returns: Partial state dict

**2. generate_scenario_node**
- Input: State with `teacher_config`
- Actions:
  - Call `generate_scenario(state["teacher_config"])` agent
  - Store result in `scenario_draft`
- Returns: `{"scenario_draft": scenario}`

**3. present_for_review**
- Input: State with `scenario_draft`
- Actions:
  - Call `interrupt()` with scenario payload:
    ```python
    interrupt({
        "type": "scenario_review",
        "scenario": state["scenario_draft"],
        "scenario_id": state["scenario_id"],
        "class_id": state["class_id"],
    })
    ```
  - After resume, extract decision from `Command(resume={...})`
  - Validate decision is "approve" (decline handled in 2.3)
- Returns: `{"approval_status": "approved"}`

### Graph Builder

```python
def create_teacher_review_graph(checkpointer=None):
    builder = StateGraph(TeacherReviewState)
    
    builder.add_node("initialize", initialize_teacher_session)
    builder.add_node("generate", generate_scenario_node)
    builder.add_node("review", present_for_review)
    
    builder.set_entry_point("initialize")
    builder.add_edge("initialize", "generate")
    builder.add_edge("generate", "review")
    
    # END after review node (single approve path in happy path)
    builder.add_edge("review", END)
    
    return builder.compile(checkpointer=checkpointer)
```

---

## Implementation Details

### 1. State Schema (`src/summit_sim/graphs/state.py`)

Add the `TeacherReviewState` TypedDict with proper type annotations and reducers.

**Key Decisions**:
- Include `feedback_history` with `append_reducer` even though unused in 2.1
- Include `retry_count` initialized to 0
- All fields type-annotated for mypy

### 2. Teacher Review Graph (`src/summit_sim/graphs/teacher_review.py`)

Create new module with:
- Node functions (async where needed)
- Graph builder function
- Import patterns matching `simulation.py`

**Patterns to Follow**:
- Use `@mlflow.trace(span_type=SpanType.AGENT)` decorator on agent calls
- Post-process deterministic fields outside LLM calls
- Clear docstrings (short summary only, no Args/Returns sections)

### 3. Tests (`tests/test_teacher_review.py`)

**Test Coverage**:
- `test_initialize_teacher_session`: Verify ID generation, state initialization
- `test_generate_scenario_node`: Mock generator agent, verify state update
- `test_present_for_review_interrupt`: Mock interrupt, verify payload structure
- `test_full_happy_path`: Integration test with mocked interrupt resume
- `test_mlflow_tags`: Verify `sme_approved: true` tag logged

**Mocking Strategy**:
- Patch `summit_sim.agents.config.Agent` (not individual files)
- Use `clear_agent_cache` fixture between tests
- Mock `interrupt()` to simulate user approval

### 4. Notebook Expansion (`notebooks/summit-sim-demo.ipynb`)

Add new section **"Phase 0: Teacher Review Flow"** before existing Phase 1:

```markdown
## Phase 0: Teacher Review Flow

Demonstrates the teacher workflow:
1. **Configuration** - Teacher sets scenario parameters
2. **Generation** - AI creates scenario from config
3. **Review** - Teacher reviews scenario via interrupt
4. **Approval** - Teacher approves scenario
5. **Share** - Unique URL generated for students
```

**Code Cells**:
1. Teacher config setup
2. Initialize and run graph to interrupt point
3. Display scenario for "review" (print formatted output)
4. Simulate approval with `Command(resume={...})`
5. Display generated URL with scenario_id and class_id
6. Pass scenario to existing Phase 1 section

### 5. Chainlit App (`src/summit_sim/app.py`)

**Entry Point Structure**:

```python
@cl.on_chat_start
async def start():
    # Show mode selection (Teacher/Student checkbox)
    # Default to Teacher mode
    
@cl.on_message
async def main(message: cl.Message):
    # Route based on current session state
```

**Teacher Flow UI**:

1. **Config Form** (on chat start):
   - Slider: Number of participants (1-20, default: 3)
   - Dropdown: Activity type (canyoneering, skiing, hiking)
   - Dropdown: Difficulty (low, med, high)
   - "Generate Scenario" button

2. **Loading State**:
   - Show spinner while graph runs to interrupt

3. **Review Screen** (at interrupt):
   - Display scenario card:
     - Title (large header)
     - Setting description
     - Patient summary
     - Learning objectives (bullet list)
     - Turns count
   - Single "Approve & Generate Link" button (green)

4. **Approval Result**:
   - Display success message
   - Show shareable URL: `/scenario/{scenario_id}?class_id={class_id}`
   - Copy-to-clipboard button

**State Management**:
- Store graph instance in `cl.user_session`
- Store graph config (with thread_id) in `cl.user_session`
- Resume graph on button click with `Command(resume={...})`

### 6. MLflow Integration

**Tracing Strategy**:
- Use existing `simulation_session()` context manager
- Wrap teacher graph execution in session
- Tag the trace with:
  - `sme_approved: "true"`
  - `class_id: {class_id}`
  - `scenario_id: {scenario_id}`
- Log params:
  - `activity_type`
  - `difficulty`
  - `num_participants`

**Verification**:
- Test that trace appears in MLflow with correct tags
- Test that scenario can be queried by class_id

---

## URL Structure

### Path-Based Routing (Chainlit)

- **Teacher Dashboard**: `/` (default entry point)
- **Scenario Review**: `/review/{scenario_id}` (redirect after generation)
- **Student Simulation**: `/scenario/{scenario_id}?class_id={class_id}`

**Link Generation**:
```python
shareable_url = f"/scenario/{scenario_id}?class_id={class_id}"
```

---

## File Changes

### New Files
```
src/summit_sim/graphs/teacher_review.py    # Graph implementation
tests/test_teacher_review.py               # Unit + integration tests
```

### Modified Files
```
src/summit_sim/graphs/state.py             # Add TeacherReviewState
src/summit_sim/app.py                      # Chainlit entry point
notebooks/summit-sim-demo.ipynb            # Add teacher flow section
```

---

## Testing Strategy

### Unit Tests

**Node Isolation Tests**:
- Mock all dependencies (agents, ID generators)
- Verify state transformations
- Verify interrupt payload structure

**Integration Test**:
- Full flow with mocked interrupt
- Mock generator to return test scenario
- Mock interrupt to simulate approval
- Verify final state has `approval_status="approved"`

### E2E Testing (Notebook)

- Run full teacher flow programmatically
- Verify scenario generated
- Verify interrupt presents reviewable data
- Verify approval produces valid URL
- Use generated scenario in subsequent simulation phase

### Quality Gates

```bash
# After implementation
ruff check --fix . && ruff format .
coverage run -m pytest tests/test_teacher_review.py && coverage report
```

---

## Dependencies

**New Dependencies**:
- `chainlit` - UI framework (likely already in pyproject.toml)

**Existing Dependencies Used**:
- `langgraph` - Graph orchestration
- `pydantic-ai` - Agent framework
- `mlflow` - Tracing and logging

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LangGraph interrupt complexity | Follow exact pattern from `simulation.py` and Epic 1 notebook |
| State management between Chainlit messages | Store graph config in `cl.user_session` |
| MLflow context propagation | Use existing `simulation_session()` wrapper |
| Interrupt payload structure changes | Version the payload type in interrupt data |

---

## Definition of Done

- [ ] All code implemented following project conventions
- [ ] Unit tests pass with ≥80% coverage
- [ ] Integration test demonstrates full happy path
- [ ] Notebook expanded with teacher flow demonstration
- [ ] Chainlit app runs without errors (manual smoke test)
- [ ] MLflow traces show correct tags and params
- [ ] Ruff linting passes with no errors
- [ ] PR reviewed and merged

---

## Future Work (Story 2.3+)

**Deferred to Story 2.3**:
- Decline button and feedback textarea
- Regeneration loop with feedback context
- Retry counter display
- Max retry fallback to golden scenario
- `feedback_history` population
- `retry_count` increment logic

**Deferred to Story 2.4**:
- Query interface for scenarios
- Export to JSON
- Advanced MLflow tagging (`golden_candidate`, etc.)

---

## References

- Epic Plan: `plans/epic-2-web-poc.md`
- Architecture: `plans/high-level-arch.md`
- Existing Simulation Graph: `src/summit_sim/graphs/simulation.py`
- State Definitions: `src/summit_sim/graphs/state.py`
- Demo Notebook: `notebooks/summit-sim-demo.ipynb`
