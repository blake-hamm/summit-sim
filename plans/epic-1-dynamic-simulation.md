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
- UI accepts free-text input (max 500 chars)
- New ActionResponder agent evaluates and generates dynamically
- Graph loops without pre-generated turn structure
- Global max_turns setting (default: 5)

---

## Pre-requisites

Complete these before starting Phase 1:

### 1. Prompt Versioning System ✅ COMPLETED
**File**: `src/summit_sim/agents/config.py`

**Implementation**:
- **System prompts**: `_get_or_register_system_prompt()` compares current system_prompt string with registered version; if different, registers new version
- **User prompts**: New `_get_or_register_user_prompt()` handles user prompt templates (Jinja2 format)
- **API Change**: `get_agent()` now accepts optional `user_prompt_template` parameter and returns `(agent, user_prompt_uri)` tuple when provided
- **Versioning**: Both use `prompts:/{agent_name}-{system|user}@latest` with automatic version updates on change detection

**Usage**:
```python
agent, user_prompt_uri = get_agent(
    agent_name="my-agent",
    output_type=MySchema,
    system_prompt=SYSTEM_PROMPT,
    user_prompt_template=USER_PROMPT_TEMPLATE,  # Optional
)
user_prompt = mlflow.genai.load_prompt(user_prompt_uri)
prompt = str(user_prompt.format(var=value))
```

**Why**: Prevents stale prompts from being used after code changes. Required before Phase 2 when ActionResponder is created.

### 2. Settings Structure ✅ COMPLETED
**File**: `src/summit_sim/settings.py`

**Implementation**:
```python
class Settings(BaseSettings):
    # ... existing settings ...
    max_turns: int = Field(default=5, description="Maximum turns per scenario")
```

**Why**: Required for Phase 2 turn limit logic.

### 3. Test Infrastructure ✅ COMPLETED
**Files**: `tests/conftest.py`, `tests/test_agents_config.py`

**Implementation**:
- **New file**: `tests/conftest.py` with reusable fixtures:
  - `mock_api_key`: Mocks OpenRouter API key
  - `clear_agent_cache`: Clears agent container between tests  
  - `mock_mlflow_prompts`: Mocks MLflow prompt loading/registration
  - `mock_agent`: Creates mock PydanticAI agent
  - `mock_generator_prompts`, `mock_simulation_prompts`, `mock_debrief_prompts`: Agent-specific prompt mocks

- **Updated**: `tests/test_agents_config.py` with comprehensive tests:
  - System prompt versioning (unchanged vs changed vs new)
  - User prompt versioning (unchanged vs changed vs new)
  - `get_agent()` with and without `user_prompt_template`

**Why**: Each phase requires E2E validation; need infrastructure ready.

---

## Pre-requisites Completion Summary

All pre-requisites have been implemented and tested:

### Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/summit_sim/agents/config.py` | Added prompt versioning for system + user prompts, @overload decorators | +84 |
| `src/summit_sim/settings.py` | Added `max_turns` setting with Field() | +4 |
| `src/summit_sim/agents/generator.py` | Updated to use new `get_agent()` API with user prompt | +5/-6 |
| `src/summit_sim/agents/simulation.py` | Updated to use new `get_agent()` API with user prompt | +5/-5 |
| `src/summit_sim/agents/debrief.py` | Updated to use new `get_agent()` API with user prompt | +6/-6 |
| `tests/conftest.py` | NEW - Reusable mock fixtures | +124 |
| `tests/test_agents_config.py` | Added comprehensive tests for prompt versioning | +77/-14 |

### Test Results
- **88 tests passing** (added 5 new tests for prompt versioning)
- All existing tests continue to pass
- Coverage maintained

### Key Implementation Details

1. **Prompt Versioning**: Uses simple string comparison (not hashing). When content differs from registered version, automatically registers new version in MLflow.

2. **get_agent() API**: 
   - Without `user_prompt_template`: returns `Agent`
   - With `user_prompt_template`: returns `(Agent, user_prompt_uri)`
   - Uses `@overload` for proper type hints

3. **User Prompt Pattern**: Agents now receive `user_prompt_uri` and call:
   ```python
   user_prompt = mlflow.genai.load_prompt(user_prompt_uri)
   prompt = str(user_prompt.format(var=value))
   ```

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
│  │     (with constraint)         │  │  May only worsen if was_correct=False
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
│  Context (all passed to agent):            │
│  - Full scenario (setting, patient_summary,│
│    hidden_truth, initial_narrative)         │
│  - Current hidden_state & scene_state      │
│  - Transcript history (past turns)         │
│                                             │
│  1. Evaluate (first in schema):            │
│     - was_correct                          │
│     - is_ending                            │
│     - feedback                             │
│                                             │
│  2. Generate narrative (second):          │
│     - Uses was_correct as context          │
│     - Constraint: worsen only if False     │
│                                             │
│  3. Evolve state:                          │
│     - updated_hidden_state                 │
│     - updated_scene_state                  │
└─────────────────────────────────────────────┘
                            │
                            ▼
                    DynamicTurnResult
                    (single LLM call)
```

### Ending Detection

- **ActionResponder's `is_ending`**: Boolean signal from the agent when scenario has reached natural conclusion (evacuation complete, patient stabilized, etc.)
- **Turn limit**: Global setting (default: 5 turns, configured in settings.py). If `turn_count >= max_turns`, force end.
- **Logic**: End if `is_ending=True` OR `turn_count >= max_turns`
- **LangGraph edge**: Conditional edge checks `is_ending` or `turn_count >= max_turns` to route to debrief or loop back

---

## Implementation Phases

### Phase 1: Author Flow - Content Creation (Testable)

**Goal**: Authors can create scenarios with `initial_narrative`, review and approve them. No student functionality yet.

**E2E Test**:
1. Start Docker container
2. Author creates scenario
3. Author sees `initial_narrative` in review screen
4. Author can rate (if < 3, regenerates - existing behavior)
5. Author approves scenario
6. Verify NO pre-generated turns in the draft

**Changes**:
1. `schemas.py`:
   - Add `DynamicTurnResult` schema
   - Update `ScenarioDraft`: add `initial_narrative: str`, make `turns` optional, add scenario-level `hidden_state` and `scene_state`

2. `agents/generator.py`:
   - Remove multi-turn generation logic
   - Output: setting, patient_summary, hidden_truth, learning_objectives, **initial_narrative**
   - Generate `initial_narrative` as the canonical first prompt

3. `ui/author.py`:
   - Display `initial_narrative` in review screen
   - Remove per-turn breakdown (no pre-generated turns)
   - Existing rating behavior applies (regenerate if rating < 3)

---

### Phase 2: Student Flow - Dynamic Simulation (Testable)

**Goal**: Students play scenarios with free-text input. AI evaluates actions and generates narrative dynamically.

**E2E Test**:
1. Start Docker container
2. Student joins via link
3. Student sees `initial_narrative`
4. Student types free-text action (max 500 chars)
5. AI evaluates and responds with feedback + narrative
6. Turn count increments
7. Repeat until `is_ending=True` or `turn_count >= max_turns`
8. Debrief shows results

**Changes**:
1. `agents/action_responder.py` (NEW):
   - System prompt: Evaluate student's free-text action medically, generate narrative, evolve state
   - Constraint: Can only worsen patient condition if `was_correct=False`
   - Input: student action + scenario context + hidden/scene state + transcript history
   - Output: `DynamicTurnResult` schema (single call)
   - Register prompt in MLflow as `prompts:/action-responder-user@latest`

2. `agents/config.py`:
   - Register ActionResponder via `get_agent()`

3. `settings.py`:
   - Add `max_turns` global config (default: 5)

4. `graphs/simulation.py`:
   - Update `SimulationState`: remove `current_turn_id`, add `turn_count`, `hidden_state`, `scene_state`, change `last_selected_choice` → `last_student_action`
   - Refactor nodes: `present_turn` → `present_prompt`, `process_player_turn` → `process_player_action`
   - New graph flow: `initialize → present_prompt → interrupt → process_player_action → update_state → check_ending → [loop or debrief]`
   - Update `TranscriptEntry`: store `student_action: str` instead of `choice_id`

5. `ui/simulation.py`:
   - Replace `cl.AskActionMessage` with `cl.AskUserMessage`
   - Prompt: initial_narrative or "What will you do?"
   - Enforce 500 character limit
   - Display feedback + narrative from ActionResponder

6. `public/hide-chat.css`:
   - Scope hiding to author mode only (body.author-mode)

7. `main.py`:
   - Set body class based on mode ("author" or "player")

---

## File Changes Summary

| File | Changes | Phase | Status |
|------|---------|-------|--------|
| `agents/config.py` | Prompt versioning for system + user prompts | Pre-req | ✅ Complete |
| `settings.py` | Add global max_turns (default: 5) | Pre-req | ✅ Complete |
| `tests/conftest.py` | Reusable mock fixtures | Pre-req | ✅ Complete |
| `schemas.py` | Add DynamicTurnResult; update ScenarioDraft | 1 | Pending |
| `agents/generator.py` | Remove multi-turn logic, output initial_narrative | 1 | Pending |
| `ui/author.py` | Display initial_narrative in review | 1 | Pending |
| `agents/action_responder.py` | NEW - ActionResponder agent | 2 | Pending |
| `agents/config.py` | Register ActionResponder | 2 | Pending |
| `graphs/simulation.py` | Refactor state, nodes, graph flow | 2 | Pending |
| `ui/simulation.py` | Replace buttons with text input | 2 | Pending |
| `public/hide-chat.css` | Scope to author mode | 2 | Pending |
| `main.py` | Set body class based on mode | 2 | Pending |

---

## Notes

- **State persistence**: `hidden_state` and `scene_state` stored in `SimulationState`, passed to ActionResponder each turn
- **Transcript history**: Full transcript passed to ActionResponder for continuity
- **Prompt versioning**: System prompts use semantic versioning; MLflow prompts use hash comparison
- **Fail fast**: Errors surface immediately - this is a breaking change, we want to know if things break
- **Validation judges**: Deferred to later epic
- **Debrief**: Will adapt existing debrief for free-text transcript
