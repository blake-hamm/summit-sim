# Story 2.1: Teacher Review Graph - Happy Path

**Epic**: Epic 2: Web POC with Teacher Review  
**Status**: ✅ Completed  
**Estimated Effort**: 1-2 days  
**Actual Effort**: ~1 day  

## Overview

Implement the teacher review workflow with human-in-the-loop (HITL) via LangGraph's `interrupt()` pattern. This is the foundation of the teacher persona flow - teachers generate scenarios, review them, approve immediately (happy path only), and receive a shareable link for students.

**Scope**: Happy path only - generate → review → approve → share. No decline/regenerate logic (deferred to Story 2.3).

---

## Success Criteria

- [x] `TeacherReviewState` defined with full schema (including future-use fields)
- [x] Teacher graph executes: initialize → generate → interrupt → approve → END
- [x] Notebook demonstrates full teacher flow end-to-end
- [x] Chainlit renders config wizard (3-step) and review screen
- [x] Unique URL generated with scenario_id and class_id
- [x] MLflow logs generation with `sme_approved: true` tag
- [x] Unit and integration tests pass (≥80% coverage)
- [x] Ruff linting passes
- [x] Message composer hidden for cleaner button-only UX
- [x] Translation files cleaned (English only)

## Running the Chainlit App

### Start the App

```bash
nix develop
uv run chainlit run src/summit_sim/app.py
```

The app will start on `http://localhost:8000`.

### Manual Testing Steps

1. **Open browser** and navigate to `http://localhost:8000`

2. **Select "I'm a Teacher"** button

3. **Configure scenario** in the form:
   - Adjust number of participants (1-20)
   - Select activity type (canyoneering/skiing/hiking)
   - Select difficulty (low/med/high)
   - Click "Generate Scenario"

4. **Wait for generation** - You'll see "Generating scenario..." while the AI creates the scenario

5. **Review the scenario** - You'll see:
   - Scenario title
   - Setting description
   - Patient summary
   - Learning objectives (bullet list)
   - Total turns count
   - Click "Approve & Generate Link"

6. **View shareable link** - After approval, you'll see:
   - Scenario ID
   - Class ID
   - Shareable URL: `/scenario/{scenario_id}?class_id={class_id}`

### What to Look For

- Config form renders with slider and dropdowns
- Loading message appears during generation
- Review screen displays all scenario details
- Approval button works and shows success message
- Shareable URL is generated with correct format
- No errors in terminal output
- Graph state is properly managed between interrupts

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

### 5. Chainlit App (`src/summit_sim/app.py`) ✅ IMPLEMENTED

**Entry Point Structure**:

```python
@cl.on_chat_start
async def start() -> None:
    """Initialize chat session and start config flow."""
    cl.user_session.set("mode", "teacher")
    await ask_num_participants()
```

**Teacher Flow UI** (3-Step Wizard with Action Buttons):

1. **Config Wizard** (sequential action buttons):
   
   **Step 1: Number of Participants** (`ask_num_participants()`)
   - Uses `cl.AskActionMessage` with buttons: 1, 2, 3, 4, 5, 6+
   - Stores value in `cl.user_session.set("num_participants", value)`
   - Proceeds to Step 2 on selection
   
   **Step 2: Activity Type** (`ask_activity_type()`)
   - Uses `cl.AskActionMessage` with buttons: Hiking, Skiing, Canyoneering
   - Stores value in `cl.user_session.set("activity_type", value)`
   - Proceeds to Step 3 on selection
   
   **Step 3: Difficulty Level** (`ask_difficulty()`)
   - Uses `cl.AskActionMessage` with buttons:
     - Low - Basic first aid
     - Medium - WFA level  
     - High - WFR level
   - Stores value in `cl.user_session.set("difficulty", value)`
   - Triggers scenario generation

2. **Loading State** (`generate_scenario()`):
   - Displays: "⏳ Generating your scenario..."
   - Creates `TeacherConfig` from session values
   - Compiles graph and stores in `cl.user_session.set("graph", graph)`
   - Invokes graph with initial state
   - On success: calls `show_review_screen()`
   - On failure: displays error and restarts wizard

3. **Review Screen** (`show_review_screen(state)`):
   - Displays scenario details via multiple `cl.Message` calls:
     - Scenario ID header
     - Title and setting
     - Patient summary
     - Learning objectives (formatted as bullet list: "• {obj}")
     - Total turns count
   - Shows `cl.AskActionMessage` with single action:
     - "✅ Approve & Generate Link" button

4. **Approval & URL Generation** (`handle_approval(state)`):
   - Retrieves graph from session
   - Resumes graph with `Command(resume={"decision": "approve"})`
   - On approval success, displays:
     - ✅ Scenario Approved! header
     - Scenario ID: `{scenario_id}`
     - Class ID: `{class_id}`
     - Shareable URL: `/scenario/{scenario_id}?class_id={class_id}`
   - On failure: shows error and returns to review screen

**State Management Patterns**:
- **Session Storage**: Uses `cl.user_session.set()` / `cl.user_session.get()` for:
  - `mode` = "teacher" (for future student mode support)
  - `num_participants`, `activity_type`, `difficulty` (config values)
  - `teacher_config` (Pydantic TeacherConfig object)
  - `graph` (compiled LangGraph instance for resume)
  - `id` (Chainlit session ID used as thread_id)

- **Graph Config**: 
  ```python
  thread_id = cl.user_session.get("id")
  config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}
  ```

- **Error Handling**: Defensive coding with fallback defaults:
  ```python
  num_participants = int(num_participants_val) if num_participants_val is not None else 3
  ```

**UI Customizations**:
- **Hidden Message Composer**: CSS/JS hides `#message-composer` for cleaner button-only UX
- **Welcome Message**: `chainlit.md` shows Summit-Sim branded intro
- **English Only**: Removed 20+ translation files, set `language = "en-US"`
- **Session Timeouts**: 1 hour session, 15 days user session

**Files Created**:
- `src/summit_sim/app.py` (282 lines) - Main application
- `chainlit.md` - Welcome message
- `public/hide-chat.css` - Hides message composer
- `public/hide-chat.js` - Backup JS for hiding composer
- `.chainlit/config.toml` - Configuration with custom CSS/JS paths

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

- [x] All code implemented following project conventions
- [x] Unit tests pass with ≥80% coverage (actual: 93%)
- [x] Integration test demonstrates full happy path
- [x] Notebook expanded with teacher flow demonstration
- [x] Chainlit app runs without errors (manual smoke test)
- [x] MLflow traces show correct tags and params
- [x] Ruff linting passes with no errors
- [x] PR reviewed and merged

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
