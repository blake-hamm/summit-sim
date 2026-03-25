# Epic 2: Web POC with Teacher Review

## Overview

Build a Chainlit web interface with two persona flows: teachers generate and review scenarios via LangGraph with HITL interrupts, students run approved scenarios. All SME feedback tracked in MLflow for golden dataset building.

**Epic Goal**: One teacher → Generate scenario → Approve in one shot → Share link → One student → Complete scenario → View debrief. Demo-ready flow.

**Architecture Principle**: Build the correct LangGraph foundation immediately (with `interrupt()`), scope initial implementation to happy path only. Add edge cases (decline/regeneration) in subsequent stories.

---

## Happy Path Flows

### Teacher Journey (Primary Flow)

```
Teacher opens app
    ↓
Fills 3-field config form (participants, activity, difficulty)
    ↓
Clicks "Generate Scenario"
    ↓
[LangGraph: generate node] → AI creates scenario
    ↓
[LangGraph: interrupt] → Presents scenario for review
    ↓
Teacher views: title, setting, patient summary, learning objectives, all turns
    ↓
Clicks "Approve Scenario" (single button, no decline in happy path)
    ↓
[LangGraph: END] → Returns approved scenario
    ↓
System displays shareable link with scenario_id and class_id
    ↓
Teacher can view student debriefs in dashboard
```

**Happy Path Outcome**: Teacher completes full flow in <2 minutes. Zero regeneration loops.

### Student Journey (Primary Flow)

```
Student clicks link with class_id
    ↓
Views scenario intro (title, setting, patient summary)
    ↓
Clicks "Start Scenario"
    ↓
[LangGraph simulation: interrupt] → Presents Turn 1 with choices
    ↓
Student selects choice → views AI feedback
    ↓
Continues through 3-5 turns
    ↓
Reaches final turn (next_turn_id=None)
    ↓
[LangGraph: generate_debrief node] → Creates DebriefReport
    ↓
Views debrief: score, pass/fail, key mistakes, teaching points
```

**Happy Path Outcome**: Student completes 3-5 turn scenario in 5-10 minutes. Single session, no persistence across browser refreshes required for POC.

---

## UI/UX Outcomes

### Teacher Interface

**1. Configuration Wizard** ✅ IMPLEMENTED
- Clean 3-step wizard with action buttons (not form fields):
  - **Step 1: Participants** - Buttons: 1, 2, 3, 4, 5, 6+
  - **Step 2: Activity** - Buttons: Hiking, Skiing, Canyoneering
  - **Step 3: Difficulty** - Buttons: Low (Basic first aid), Medium (WFA level), High (WFR level)
- Uses `cl.AskActionMessage` for each step
- Sequential flow: participants → activity → difficulty
- Loading state: "⏳ Generating your scenario..."
- Message composer hidden for cleaner button-only UX

**2. Review Screen** ✅ IMPLEMENTED
- Scenario display with:
  - Scenario ID header
  - Title and setting description
  - Patient summary
  - Learning objectives (formatted bullet list)
  - Total turns count
- Single "✅ Approve & Generate Link" action button
- After approval, displays:
  - Scenario ID
  - Class ID (internal use, not in URL)
  - Shareable URL: `?scenario_id={scenario_id}` (uses BASE_URL env var)
- Note: Hidden truth not displayed in current implementation

**3. Teacher Dashboard** (post-approval)
- List of scenarios created by this session
- Each card shows: class_id, activity type, approval status
- Click to view student transcripts and debriefs

### Student Interface

**1. Join Screen** ✅ IMPLEMENTED
- Scenario ID auto-populated from URL param (`?scenario_id=`)
- Scenario intro card (title, setting, patient, learning objectives)
- "Start Scenario" button to begin

**2. Turn Display** (interrupt payload)
- Narrative text in chat bubble style
- Scene state context (weather, time, conditions)
- Choice buttons (2-3 options, large click targets)
- Choice descriptions as button labels

**3. Feedback Display**
- AI feedback in assistant message bubble
- Learning moments as info callouts
- "Continue" button to advance

**4. Debrief Screen**
- Score percentage (large, color-coded: green ≥70%, red <70%)
- Pass/fail badge
- Summary paragraph
- Key mistakes (bullet list with warning icons)
- Strong actions (bullet list with check icons)
- Teaching points (info cards)
- Recommendations for next time

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Chainlit UI                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Teacher   │  │   Teacher    │  │     Student      │   │
│  │   Config    │  │   Review     │  │   Simulation     │   │
│  │    Form     │  │   Screen     │  │     Flow         │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
└─────────┼────────────────┼───────────────────┼─────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph Workflows                     │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │      Teacher Review Graph           │                   │
│  │                                     │                   │
│  │  ┌─────────┐    ┌──────────────┐   │                   │
│  │  │generate │───▶│   interrupt  │   │  [Happy Path]     │
│  │  │scenario │    │(present review)│  │                   │
│  │  └─────────┘    └──────┬───────┘   │                   │
│  │                        │           │                   │
│  │              ┌─────────┴─────────┐ │                   │
│  │              │ Command(resume=   │ │                   │
│  │              │   {"decision":    │ │                   │
│  │              │    "approve"})    │ │                   │
│  │              └─────────┬─────────┘ │                   │
│  │                        ▼           │                   │
│  │                     [END]          │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │      Student Simulation Graph       │                   │
│  │        (from Epic 1)                │                   │
│  │                                     │                   │
│  │  initialize → present → process →   │                   │
│  │    update → check → [loop/END]      │                   │
│  │                                     │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     MLflow Tracing                          │
│                                                             │
│  Teacher Generation Run:                                    │
│    ├── tags: class_id, sme_approved: true                  │
│    └── params: num_participants, activity, difficulty      │
│                                                             │
│  Student Simulation Run:                                    │
│    ├── tags: class_id, scenario_id                         │
│    └── metrics: final_score, completion_status             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

**1. LangGraph with interrupt()**
- Teacher and student flows both use `interrupt()` for HITL
- Pattern established in Epic 1 simulation graph
- Checkpointer (InMemorySaver for POC) preserves state
- Resume via `Command(resume={...})`

**2. Unified State Management**
- `TeacherReviewState` (TypedDict):
  - `teacher_config: TeacherConfig`
  - `scenario_draft: ScenarioDraft | None`
  - `scenario_id: str`
  - `class_id: str`
  - `retry_count: int` (0 in happy path, used later)
  - `feedback_history: list[str]` (empty in happy path)
  - `approval_status: str | None`

**3. Chainlit Integration Pattern**
- Single entry point at `/` with mode selection (Teacher/Student checkbox)
- Graph state stored in `cl.user_session` between interrupts
- Each interrupt displays appropriate UI elements
- Button callbacks resume graph with `Command(resume=...)`
- URL structure:
  - Teacher dashboard: `/`
  - Scenario review: `/review/{scenario_id}`
  - Student simulation: `/scenario/{scenario_id}?class_id={class_id}`
- No persistence across browser sessions (for POC)

**4. MLflow Tracing**
- Single parent run per scenario
- Child spans for each generation/simulation step
- Tags link teacher generation to student runs via class_id
- Approved scenarios tagged `sme_approved: true` in 2.1
- `golden_candidate` and `needs_review` tags added in 2.4

**5. Storage Strategy**
- LangGraph `InMemorySaver` for session state (checkpoints)
- LangGraph `InMemoryStore` for scenario storage (shared between teacher/student flows)
- `scenario_store` in `graphs/utils.py` - singleton instance
- MLflow for trace logging and metadata storage
- No persistence across app restarts (InMemoryStore cleared on restart)
- Scenarios saved to store on teacher approval, retrieved by student via scenario_id

---

## Story Breakdown

### Story 2.1: Teacher Review Graph - Happy Path ✅ COMPLETED

**Goal**: Teacher generates scenario, reviews it, approves immediately. Single attempt, no edge cases.

**Status**: Fully implemented and tested. All quality gates passing.

**Architecture**:
```
┌─────────┐    ┌──────────────┐    ┌─────────┐
│initialize│───▶│   generate   │───▶│interrupt│
└─────────┘    │  scenario    │    │(review) │
               └──────────────┘    └────┬────┘
                                        │
                     Command(resume={"decision": "approve"})
                                        │
                                        ▼
                                     [END]
```

**Nodes**:
- `initialize_teacher_session`: Create state, generate IDs
- `generate_scenario_node`: Call generator agent
- `present_for_review`: Interrupt with scenario payload

**State Fields**:
- All standard fields initialized
- `retry_count=0` (not used yet, but present)
- `approval_status=None` → "approved" after resume

**UI Implementation** (Chainlit):
- **3-step wizard**: Participants (1-6+) → Activity (hiking/skiing/canyoneering) → Difficulty (low/med/high)
- Each step uses `cl.AskActionMessage` with action buttons
- Loading state: "⏳ Generating your scenario..."
- Review screen displays: title, setting, patient summary, learning objectives (bullets), total turns
- Single "✅ Approve & Generate Link" button
- Shareable URL format: `/scenario/{scenario_id}?class_id={class_id}`
- Message composer hidden via CSS/JS for cleaner UX

**Files Created**:
- `src/summit_sim/app.py` - Main Chainlit application
- `public/hide-chat.css` - Hides message composer
- `public/hide-chat.js` - Backup JS hiding
- `chainlit.md` - Welcome message
- `.chainlit/config.toml` - Configuration with custom CSS/JS paths

**Excludes** (deferred to Story 2.3):
- Decline button
- Feedback textarea
- Retry logic
- Max retry fallback

**TDD**:
- ✅ Test: Config → generate → interrupt → approve → END
- ✅ Verify: MLflow shows generation trace with approval tag
- ✅ Coverage: 93% (exceeds 80% requirement)

**Implementation Details**: See `plans/story-2-1-teacher-review-happy-path.md` for full implementation guide.

---

### Story 2.2: Student Flow - Happy Path ✅ COMPLETED

**Goal**: Student joins via link, runs scenario, sees debrief. Reuses Epic 1 simulation graph.

**Status**: Fully implemented and tested.

**Architecture**: Use existing `create_simulation_graph()` from `src/summit_sim/graphs/simulation.py`

**Flow**:
- Student joins with class_id
- Load scenario from checkpoint/state
- Run simulation graph with interrupts for choices
- Display debrief from `state["debrief_report"]`

**UI Elements**:
- Join screen with class_id input
- Scenario intro display
- Turn presentation (narrative + choice buttons)
- Feedback display after each choice
- Debrief screen with full report

**Excludes**:
- Multi-student concurrent sessions
- Session persistence across refreshes
- "Resume where I left off"

**TDD**:
- Test: Join → 3 turns → debrief displays
- Verify: MLflow links to teacher's generation via class_id

---

### Story 2.3: Teacher Edge Cases - Decline & Regenerate

**Goal**: Add decline/regeneration loop to existing teacher graph.

**Architecture** (extends 2.1):
```
                    ┌──────────────────┐
                    │   decline path   │
                    │  (new in 2.3)    │
                    └────────┬─────────┘
                             │
    ┌─────────┐    ┌─────────┴─────────┐    ┌─────────┐
    │interrupt│───▶│  process_decision │───▶│check_   │
    │(review) │    │                   │    │decision │
    └────┬────┘    └───────────────────┘    └────┬────┘
         │                                        │
         │         approve                        │ decline + retry < 3
         │            │                           │
         │            ▼                           ▼
         │         [END]                    ┌──────────┐
         │                                  │regenerate│
         │                                  │(feedback)│
         │                                  └────┬─────┘
         │                                       │
         └───────────────────────────────────────┘
                         (loop back to interrupt)
```

**New Nodes**:
- `process_decision`: Parse approve/decline + feedback
- `check_decision`: Route to END, regenerate, or fallback
- `regenerate_with_feedback`: Call generator with context

**UI Additions**:
- "Decline & Provide Feedback" button
- Feedback textarea
- Retry counter display ("Attempt 2 of 3")
- Fallback message on max retries

**State Additions**:
- `feedback_history: append_reducer` list
- `retry_count` increments on decline
- `last_feedback: str`

**MLflow**:
- Log declined attempts with `sme_approved: false`
- Log teacher feedback as param
- Tag final approved with `retry_count`

**TDD**:
- Test: Decline → regenerate → approve
- Test: 3 declines → fallback to golden scenario
- Verify: MLflow shows all attempts with feedback

---

### Story 2.4: Observability & Golden Dataset

**Goal**: Track all scenarios in MLflow for SME feedback analysis.

**Features**:
- Tag approved scenarios: `golden_candidate: true`
- Tag declined scenarios: `needs_review: true`
- Query interface: list scenarios by status, date range
- Export function: approved scenarios → JSON dataset

**MLflow Schema**:
```python
# Teacher Generation Run (Story 2.1)
tags = {
    "class_id": "abc123",
    "scenario_id": "scn-a3f8d2e9",
    "sme_approved": "true",  # "true" in 2.1 happy path
}
params = {
    "activity_type": "hiking",
    "difficulty": "med",
    "num_participants": "4",
}

# Teacher Generation Run (Story 2.4 - adds:)
tags = {
    "golden_candidate": "true",  # Approved scenarios
    "needs_review": "true",      # Declined scenarios
    "retry_count": "0",
}
params = {
    "teacher_feedback": "",  # Populated in 2.3+
}

# Student Simulation Run
tags = {
    "class_id": "abc123",
    "scenario_id": "scn-a3f8d2e9",
}
metrics = {
    "final_score": 75.0,
    "completion_status": 1.0,  # 1=pass, 0=fail
}
```

**TDD**:
- Test: Query MLflow for approved scenarios
- Test: Export scenario to JSON format

---

### Story 2.5: Multi-Student Support (Post-POC)

**Goal**: Multiple students can join same class_id independently.

**Requirements**:
- DragonflyDB persistence (replace InMemorySaver)
- Session isolation per student
- Independent checkpoint streams
- Shared scenario state, separate transcript states

**Note**: Explicitly deferred from initial POC. Single student per scenario for Stories 2.1-2.4.

---

## Development Sequence

### Iteration 1: Foundation (Stories 2.1 + 2.2)
**Goal**: Working happy path end-to-end

**Deliverables**:
- TeacherReviewState schema
- Teacher review graph (3 nodes, 2 edges)
- Chainlit teacher UI (config form → review → approval)
- Chainlit student UI (join → simulate → debrief)
- MLflow traces for both flows

**Demo**: Teacher generates → approves → shares link → student completes → debrief shown

### Iteration 2: Robustness (Story 2.3)
**Goal**: Handle edge cases

**Deliverables**:
- Decline/regeneration loop
- Feedback context in regeneration
- Max retry handling (3 attempts)
- Golden scenario fallback

### Iteration 3: Observability (Story 2.4)
**Goal**: Dataset building

**Deliverables**:
- MLflow tagging complete
- Scenario query interface
- Export to JSON

### Iteration 4: Scale (Story 2.5)
**Goal**: Production readiness

**Deliverables**:
- DragonflyDB persistence
- Multi-student concurrent sessions
- Session recovery

---

## Implementation Notes

### Story 2.1 Implementation Decisions

**UI Pattern: Action Buttons vs Form Fields**
- **Decision**: Used 3-step wizard with `cl.AskActionMessage` buttons instead of traditional form inputs
- **Rationale**: Cleaner UX, no free-form text entry needed, reduces user errors
- **Trade-off**: Limited to predefined options (1-6+ participants instead of 1-20 slider)

**Message Composer Hiding**
- **Problem**: Chainlit shows a message composer (text input) by default, which conflicts with button-based interaction
- **Solution**: Custom CSS (`public/hide-chat.css`) and JavaScript (`public/hide-chat.js`) to hide `#message-composer`
- **Result**: Clean button-only interface for the wizard flow

**Translation Cleanup**
- **Problem**: Chainlit includes 20+ language translation files by default
- **Solution**: Removed all non-English translations, set `language = "en-US"` in config
- **Result**: Reduced noise, focused on English-only for hackathon context

**Session State Management**
- **Pattern**: Store intermediate config values in `cl.user_session` between wizard steps
- **Graph Storage**: Store compiled LangGraph instance in session to resume after interrupt
- **Thread ID**: Use Chainlit's session ID as LangGraph thread_id for checkpointing

**Error Handling Strategy**
- **Approach**: Defensive coding with fallback defaults for all session values
- **Recovery**: On generation failure, show error message and restart wizard from beginning
- **UX**: Friendly error messages with emojis (❌) for visibility

### Current Status (Epic 2 - All Stories Complete)
- ✅ Story 2.1: Teacher flow (config → generate → review → approve)
- ✅ Story 2.2: Student flow (join → simulate → debrief)
- ✅ All tests passing, 80%+ coverage
- ✅ Ruff linting clean
- ✅ Chainlit UI functional with hidden composer
- ✅ MLflow integration working
- ✅ Shared scenario storage via LangGraph InMemoryStore
- ✅ Query param routing (?scenario_id=)
- ✅ BASE_URL env var for configurable shareable links

### Refactoring Notes (Post-Implementation)
- Renamed `app/` directory to `ui/` for clarity (was misleading)
- Renamed `app.py` to `main.py` as conventional entry point
- Merged `__init__.py` handlers directly into `main.py`
- Merged `storage.py` into `graphs/utils.py`
- Changed URL routing from path-based (`/scenario/{id}`) to query params (`?scenario_id=`)
- Removed `class_id` from shareable URL (kept in schema for future analytics use)

---

## Success Criteria

### Epic Complete When:
- [x] Teacher can complete happy path in <2 minutes
- [x] Student can complete scenario in 5-10 minutes
- [x] MLflow shows complete trace: generation → approval → student run → debrief
- [x] Demo works end-to-end without code changes
- [x] Ruff linting passes
- [x] Tests cover happy paths (≥80% coverage)

### Story 2.1 Complete When:
- [x] TeacherReviewState defined with full schema
- [x] Teacher graph runs: initialize → generate → interrupt → approve → END
- [x] Notebook expanded with teacher flow demonstration
- [x] Chainlit renders config wizard and review screen
- [x] scenario_id and class_id generated and displayed
- [x] Unique URL format: ?scenario_id={scenario_id} (uses BASE_URL env var)
- [x] MLflow logs generation with sme_approved tag
- [x] Message composer hidden for cleaner UX
- [x] Translation files cleaned (English only)

### Story 2.2 Complete When:
- [x] Student joins via ?scenario_id= URL param
- [x] Simulation graph runs with interrupts
- [x] Debrief displays from state["debrief_report"]
- [x] MLflow links student run to scenario_id

---

## Technical Notes

### Pattern Consistency
- Teacher graph mirrors student simulation graph pattern
- Both use `interrupt()` + `Command(resume=...)`
- Both use same checkpointer type (InMemorySaver for POC)
- Both log to MLflow with class_id linkage

### File Structure
```
src/summit_sim/
  main.py                   # Chainlit entry point
  settings.py               # Configuration (includes BASE_URL)
  schemas.py                # Data models
  ui/
    teacher.py              # Teacher flow handlers
    student.py              # Student flow handlers
  graphs/
    teacher_review.py      # Teacher review LangGraph
    simulation.py          # Student simulation LangGraph
    utils.py               # Shared utilities (InMemoryStore, TranscriptEntry)
  agents/
    config.py              # Agent factory
    generator.py           # Scenario generation agent
    simulation.py          # Feedback generation agent
    debrief.py             # Debrief generation agent

tests/
  test_teacher_review.py   # Teacher flow tests
  test_student_flow.py     # Student flow tests
  test_debrief.py          # Debrief agent tests

plans/
  epic-2-web-poc.md                         # This document
  story-2-1-teacher-review-happy-path.md    # Detailed implementation guide
```

### Dependencies
- `chainlit` - UI framework
- Existing: `langgraph`, `pydantic-ai`, `mlflow`

### Risk Mitigation
- **Risk**: LangGraph interrupt complexity in Chainlit
  - **Mitigation**: Follow exact pattern from Epic 1 notebook
- **Risk**: State management between interrupts
  - **Mitigation**: Store graph config in `cl.user_session`
- **Risk**: MLflow context propagation
  - **Mitigation**: Use existing tracing.py patterns

---

## Open Questions for Future Refinement

1. **Persistence**: When to add DragonflyDB for session recovery?
2. **Validation**: When to add safety/realism judges vs SME review?
3. **UI Polish**: Animation, progress indicators, mobile optimization?
4. **Multi-teacher**: Support multiple teachers with scenario libraries?
5. **Analytics**: Dashboard for scenario performance metrics?

These are intentionally out of scope for this POC-focused epic.
