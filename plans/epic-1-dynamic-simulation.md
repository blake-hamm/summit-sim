# Epic 1: Dynamic Open-Ended Simulation Engine

## Overview

Replace rigid multiple-choice turns with an interactive, free-text simulation loop that reacts to student decisions in real-time. This is the primary "wow" factor for judges - it proves the AI is doing complex medical evaluation rather than just navigating a static JSON tree.

**Epic Goal**: Student types free-text actions (e.g., "I apply a tourniquet") → AI evaluates action and generates the next narrative turn dynamically.

**Architecture Principle**: Use a single PydanticAI agent (ActionResponder) that returns a strict `DynamicTurnResult` schema. Schema field order enforces evaluation → narrative generation. LangGraph manages the outer game loop only.

---

## Current State (Pre-Epic)

### What's Currently in Place
- **Generator Agent**: Outputs full branching scenario with 3-5 pre-written turns, each with 3-5 choice buttons
- **Student UI**: Uses `AskActionMessage` (buttons), NOT free text
- **Simulation Graph**: Pre-determined flow via `next_turn_id` lookup
- **Schemas**: `ScenarioDraft` contains `turns: list[ScenarioTurn]`, each with `choices: list[ChoiceOption]`
- **Hidden/Scene State**: Static per-turn, never dynamically updated
- **Ending**: Determined by `next_turn_id=None` in pre-generated choices

### What Needs to Change
- Generator outputs only initial scenario (Setting, Patient Summary, Hidden Truth) + `initial_narrative`
- UI accepts free-text input
- Two new agents evaluate and generate dynamically
- Graph loops without pre-generated turn structure

---

## Architecture

### New Flow

```
Student types action ("I check for pulse...")
    │
    ▼
┌─────────────────────────────────────┐
│      ActionResponder               │  ← Single PydanticAI agent
│                                     │     Strict DynamicTurnResult schema
│  ┌───────────────────────────────┐  │
│  │  1. Evaluate: was_correct,   │  │  ← Evaluates action first
│  │     is_ending, feedback       │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  2. Generate: narrative_text │  │  ← Uses was_correct as context
│  │     (with constraint)         │  │     May only worsen if was_correct=False
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  3. Evolve: hidden_state,    │  │  ← Updates both state dicts
│  │     scene_state              │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
AI returns: DynamicTurnResult (single LLM call)
    - was_correct, is_ending, feedback
    - narrative_text
    - updated_hidden_state, updated_scene_state
```

### Agent Responsibilities

| Agent | Responsibility | Output Schema |
|-------|---------------|---------------|
| `ActionResponder` | Single agent: evaluates action, generates narrative, evolves state. Schema enforces evaluation → narrative order | `DynamicTurnResult` |

### Data Flow

```
Student Action Text
    │
    ▼
┌─────────────────────────────────────────────┐
│            ActionResponder                 │
│  Single PydanticAI agent                    │
│                                             │
│  1. Evaluate (first in schema):             │
│     - was_correct                           │
│     - is_ending                             │
│     - feedback                              │
│                                             │
│  2. Generate narrative (second):           │
│     - Uses was_correct as context           │
│     - Constraint: worsen only if False     │
│                                             │
│  3. Evolve state:                          │
│     - updated_hidden_state                  │
│     - updated_scene_state                   │
└─────────────────────────────────────────────┘
                            │
                            ▼
                    DynamicTurnResult
                    (single LLM call)
```

### Ending Detection

- **ActionResponder's `is_ending`**: Boolean signal from the agent when scenario has reached natural conclusion (evacuation complete, patient stabilized, etc.)
- **Turn limit**: Configurable (default: 5 turns). If `turn_count >= max_turns`, force end.
- **Logic**: End if `is_ending=True` OR `turn_count >= max_turns`
- **LangGraph edge**: Conditional edge checks `is_ending` or `turn_count >= max_turns` to route to debrief or loop back

---

## Implementation Phases

### Phase 1: Schema & Generator Updates

**Goal**: Restrict generator to output only initial scenario (no pre-generated turns).

**Changes**:
1. Add new schema to `schemas.py`:
   ```python
   class DynamicTurnResult(BaseModel):
       """Output from ActionResponder agent (single LLM call)."""
       was_correct: bool = Field(description="Evaluate if the student's medical action was correct based on the hidden truth.")
       confidence: float = Field(ge=0.0, le=1.0, description="Confidence level in the evaluation (0.0-1.0).")
       is_ending: bool = Field(description="Set to true if the scenario has reached a natural conclusion or the patient is stabilized.")
       feedback: str = Field(description="Private medical feedback for the student's action.")
       narrative_text: str = Field(description="The next scene narrative. CONSTRAINT: You may only worsen the patient's condition if was_correct is False.")
       updated_hidden_state: dict[str, str] = Field(description="Updated underlying medical truth.")
       updated_scene_state: dict[str, str] = Field(description="Updated visible scene conditions.")
   ```

2. Update `ScenarioDraft` schema:
   - Add `initial_narrative: str` field (the first "What will you do?" prompt)
   - Make `turns: list[ScenarioTurn]` optional (empty by default)
   - Move `hidden_state` and `scene_state` to scenario level

3. Update Generator Agent (`agents/generator.py`):
   - Remove multi-turn generation logic
   - Output: setting, patient_summary, hidden_truth, **initial_narrative** (the first "What will you do?" prompt)
   - Generate `initial_narrative` based on the scenario setup - this is the canonical first prompt
   - **Important**: `initial_narrative` is generated by the generator and shown to the author during review. This allows the author to see and approve the first prompt before students see it. The author can request regeneration if the initial_narrative is unsatisfactory.

**E2E Test**: Author creates scenario → reviews approval screen → sees initial_narrative displayed for review → author approves → NO pre-generated turns

---

### Phase 2: ActionResponder Agent

**Goal**: Create a single PydanticAI agent that evaluates student action, generates narrative, and evolves state in one LLM call.

**Why single agent**: Schema field order enforces evaluation → narrative. Placing evaluation fields before narrative fields forces the LLM to decide on medical correctness first, which it then uses as context to generate the narrative. This avoids the latency of sequential calls and eliminates race conditions.

**Changes**:
1. Create `agents/action_responder.py`:
   - System prompt: Evaluate student's free-text action medically, generate narrative response, evolve hidden/scene state
   - **Constraint**: Can only worsen patient condition (e.g., change patient_alive from true to false) if `was_correct=False`. If the action was correct, patient condition should not worsen.
   - Input: student action text + scenario context (setting, patient_summary, hidden_truth) + current hidden/scene state
   - Output: `DynamicTurnResult` schema (single call)

2. Register prompt template in MLflow as `prompts:/action-responder-user@latest`

**Schema Output** (field order matters):
```python
class DynamicTurnResult(BaseModel):
    # 1. Evaluate first (model generates these first)
    was_correct: bool = Field(description="Evaluate if the student's medical action was correct based on the hidden truth.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level in the evaluation (0.0-1.0).")
    is_ending: bool = Field(description="Set to true if the scenario has reached a natural conclusion or the patient is stabilized.")
    feedback: str = Field(description="Private medical feedback for the student's action.")
    
    # 2. Generate narrative based on the evaluation above
    narrative_text: str = Field(description="The next scene narrative. CONSTRAINT: You may only worsen the patient's condition if was_correct is False.")
    
    # 3. Evolve state
    updated_hidden_state: dict[str, str] = Field(description="Updated underlying medical truth.")
    updated_scene_state: dict[str, str] = Field(description="Updated visible scene conditions.")
```

**System Prompt Emphasis**: In addition to the schema description, explicitly state the constraint in the ActionResponder's system prompt:
> "CRITICAL CONSTRAINT: You may only worsen the patient's condition (e.g., change patient_alive from true to false, increase blood loss, etc.) if and only if was_correct=False. If the student's action was correct, the patient condition must not worsen as a direct result of that action."

**Error Handling**:
   - Use PydanticAI's built-in retry
   - If agent fails, LangGraph catches and returns friendly error message

**E2E Test**: Mock LLM → call ActionResponder with "I apply direct pressure" + state → returns DynamicTurnResult with was_correct=True, narrative_text, updated state

---

### Phase 3: Initial Narrative Flow

**Goal**: Ensure initial_narrative from Generator flows to student graph.

**Clarification**: The `initial_narrative` is generated in Phase 1 by the Generator agent and displayed to the author during review. The author sees and approves it as part of the scenario. After approval, it flows to the student graph.

**Changes**:
1. Verify Generator outputs `initial_narrative` in Phase 1
2. Ensure `initial_narrative` is stored in scenario state and flows to SimulationState
3. Simulation graph uses `initial_narrative` as the first prompt (not a "START" action call)

**E2E Test**: Author creates scenario → reviews and sees initial_narrative → approves → student graph loads → initial_narrative is displayed

---

### Phase 4: Simulation Graph Refactor

**Goal**: Refactor graph to use dynamic turn generation instead of turn lookup.

**Changes**:
1. Update `SimulationState` in `graphs/simulation.py`:
   - Remove `current_turn_id` (no more turn lookup)
   - Add `turn_count: int` (for 5-turn limit)
   - Add `hidden_state: dict[str, str]` (scenario-level, evolves)
   - Add `scene_state: dict[str, str]` (scenario-level, evolves)
   - Change `last_selected_choice` → `last_student_action: str`
   - Add `max_turns: int` (configurable, default 5)

2. Refactor nodes:
   - `present_turn` → `present_prompt`: Display current narrative via interrupt
   - `process_player_turn` → `process_player_action`: Call ActionResponder
   - `update_simulation_state`: Update transcript with free-text action, evolve state
   - `check_ending`: Evaluate ActionResponder's `is_ending` + turn_count >= max_turns
   - Add error handling: If orchestrator fails, catch and present friendly message to student

3. New graph flow:
   ```
   initialize → present_prompt → interrupt(text input) → 
   process_player_action → update_state → check_ending → 
   [loop to present_prompt or generate_debrief]
   ```

4. Transcript Entry updates:
   - Store `student_action: str` instead of `choice_id`
   - Store `was_correct: bool` from evaluation

**E2E Test**: Student types action → graph processes → new narrative displayed → turn_count increments

---

### Phase 5: UI Updates

**Goal**: Replace button-based input with free-text input.

**Changes**:
1. Update `public/hide-chat.css`:
   - Scope hiding to author mode only using body class:
   ```css
   body.author-mode #message-composer,
   body.author-mode #message-composer * {
       display: none !important;
   }
   ```

2. Update `main.py`:
   - Set body class based on mode ("author" or "player")

3. Update `ui/simulation.py`:
   - Replace `cl.AskActionMessage` (buttons) with `cl.AskUserMessage` (text input)
   - Prompt: "What will you do?" or dynamic initial_narrative
   - Handle free-text response

4. Update feedback display:
   - Show `was_correct` + `feedback` from ActionResponder
   - Show new `narrative_text` from ActionResponder

**E2E Test**: Student sees "What will you do?" → types "I check for breathing" → sees AI feedback → sees next narrative

---

## File Changes Summary

| File | Changes |
|------|---------|
| `schemas.py` | Add DynamicTurnResult; update ScenarioDraft |
| `agents/generator.py` | Remove multi-turn logic, output initial_narrative |
| `agents/action_responder.py` | NEW - ActionResponder agent (single PydanticAI call) |
| `graphs/simulation.py` | Refactor state, nodes, ending logic, error handling |
| `ui/simulation.py` | Replace buttons with text input |
| `agents/config.py` | Register ActionResponder via get_agent() |
| `public/hide-chat.css` | Scope hiding to author mode only |
| `src/summit_sim/main.py` | Set body class based on mode |

---

## E2E Test Checklist

| Phase | Test |
|-------|------|
| 1 | Author creates scenario → sees initial_narrative in review → author can request regeneration if unsatisfactory → NO pre-generated turns |
| 2 | ActionResponder evaluates "I apply direct pressure" → returns was_correct=True, confidence, narrative_text, respects was_correct constraint |
| 3 | After approval, initial_narrative flows to student graph |
| 4 | Graph loops without pre-generated turns, turn_count increments |
| 5a | Chat input hidden in author mode, visible in player mode |
| 5b | Student types free text → AI responds dynamically |

---

## Acceptance Criteria

- [ ] Student can type free-text actions (e.g., "I apply a tourniquet")
- [ ] ActionResponder evaluates action (returns was_correct and confidence)
- [ ] ActionResponder generates narrative (respects was_correct constraint - worsen only if was_correct=False)
- [ ] ActionResponder makes `is_ending` judgement
- [ ] Hidden state and scene state evolve dynamically
- [ ] Scenario ends early if `is_ending=True`
- [ ] Scenario ends after max_turns (configurable, default 5)
- [ ] Debrief adapted for free-text transcript
- [ ] Ruff linting passes
- [ ] Coverage ≥80%

---

## Open Questions

1. **Debrief adaptation**: Will adapt existing debrief for free-text transcript. The transcript format will change from `choice_id` to `student_action: str`, so debrief needs updating.

2. **State evolution limits**: **RESOLVED** - ActionResponder can only worsen patient condition (e.g., change patient_alive from true to false) if `was_correct=False`. This constraint is enforced in the schema description.

3. **Validation judges**: Deferred. Will create a separate judge agent later in the project for safety/realism validation. Focus on core functionality first.
