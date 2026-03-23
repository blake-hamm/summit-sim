# Summit Sim Architecture

## Overview

Summit Sim is an AI-powered wilderness rescue training simulator for collaborative learning.

The app has one main flow:
1. Host provides minimal config and triggers scenario generation.
2. AI generates complete scenario with all turns and choices.
3. System validates for medical accuracy and realism.
4. Host reviews the scenario asynchronously (accept/decline).
5. Students join via shared link; one lead student makes choices, others observe.
6. AI provides personalized feedback on each choice.
7. System produces a learning-focused debrief (pass/fail, not graded).

**Architecture**: Hybrid approach - AI generates complete scenarios upfront with pre-written turns and multiple choice options. During simulation, students select from choices and AI provides personalized feedback.

This project uses:
- **Chainlit** for the chat-based frontend and session handling
- **LangGraph** for orchestration, routing, retries, and shared state
- **DragonflyDB** (Redis-compatible) for persistence via LangGraph checkpointing
- **PydanticAI** for agent calls with structured outputs
- **OpenRouter** for multi-model LLM access via pydantic-ai's native provider
- **MLflow** for traces, golden scenarios, A/B testing, and eval logging
- **YAML config** for model assignments (per-agent and default)

## Stack

### Frontend
- **Chainlit**
- Used for:
  - Host configuration UI (3 simple fields)
  - Host async review interface (accept/decline scenarios)
  - Student chat interface with shared room links
  - Multiple choice selection display
  - Status/loading messages
  - End-of-scenario debrief view

### Workflow & State
- **LangGraph** with **DragonflyDB** persistence
- Used for:
  - Scenario generation workflow with checkpointing
  - Validation loop with retry logic
  - Interactive simulation with choice tracking
  - Shared scenario state across users
  - Routing to debrief when scenario ends

### Agents
- **PydanticAI** via **OpenRouter** native provider
- Used for:
  - Scenario Generator (creates complete scenarios with all turns)
  - Safety Judge (medical accuracy validation)
  - Realism Judge
  - Pedagogy Judge
  - Refiner
  - Simulation Feedback Agent (personalized feedback on choices)
  - Debrief Agent
- Model assignments defined in `config/models.yaml`

### Observability
- **MLflow**
- Used for:
  - Trace logging and run tracking
  - Golden scenario storage and retrieval
  - A/B testing support for model evaluation
  - Simple eval metrics
  - Debugging failed generations

## Core Components

### 1. Host Configuration
Minimal host input (3 fields):
- number of participants (1-20)
- activity type (canyoneering, skiing, hiking)
- difficulty level (low, med, high)

AI expands these into a complete wilderness rescue scenario with multiple turns.

Generates shareable room link for students to join.

**Tool used:** Chainlit

### 2. Scenario Generator
AI generates complete scenario from minimal host input:
- Creates 3-5 pre-written turns
- Each turn has 2-3 multiple choice options
- One medically optimal choice per turn
- Branches based on choices via next_turn_id

**Tool used:** PydanticAI inside a LangGraph node

### 3. Validation Loop
Runs judges to ensure medical accuracy and realism:
- safety (medical accuracy - prevents hallucinated scenarios)
- realism (environmental and situational plausibility)
- pedagogy (learning value assessment)

If validation fails, a refiner updates the scenario and judges re-run.

**Tools used:** LangGraph for orchestration, PydanticAI for each judge/refiner

### 4. Host Review
Host asynchronously reviews the validated scenario:
- Accept: scenario becomes joinable via shared link
- Decline: triggers automatic regeneration

**Tool used:** Chainlit

### 5. Simulation Engine
Runs the live scenario once accepted by host.

One lead student drives actions (others observe). The engine:
- presents current turn with multiple choice options
- accepts student's choice selection
- AI generates personalized feedback on the choice
- advances to next turn based on choice's next_turn_id
- tracks learning moments

**Tools used:** LangGraph + PydanticAI

### 6. Debrief
Learning-focused summary (pass/fail, not graded):
- what the student did well
- mistakes and teaching moments
- better actions for next time
- key learning points

**Tools used:** PydanticAI, MLflow for logged evaluation data

## Shared State

Use one shared graph state object for the whole scenario, persisted via LangGraph to DragonflyDB.

Example fields:

```python
class AppState:
    room_id                          # Shareable link code (e.g., "abc123")
    host_config                      # Minimal scenario parameters (3 fields)
    scenario_draft                   # Complete generated scenario
    validated_scenario
    host_review_status               # "pending" | "accepted" | "declined"
    current_turn_id                  # Track which turn student is on
    transcript                       # List of choices made
    last_student_choice              # Most recent choice
    validation_results
    retry_count
    scenario_status                  # "generating" | "review_pending" | "active" | "complete"
    completion_status                # "incomplete" | "pass" | "fail" (learning-focused)
    key_learning_moments
    model_config                     # Which models were used (for A/B testing)
```

### Notes
- Scenarios are pre-generated with all turns (not dynamically generated per action).
- Students select from multiple choice options (not free-form input).
- AI provides personalized feedback on choices (dynamic component).
- Make all agent outputs structured and typed.
- Add a max retry count for validation before falling back to golden scenario.
- Store golden scenarios in MLflow for demo reliability and few-shot prompting.

## Flow

### High-level pseudocode

```python
on_host_start():
    config = get_host_config()                          # Chainlit - 3 fields
    state.host_config = config
    state.room_id = generate_room_code()                # Short shareable link

    state = run_generation_flow(state)                 # LangGraph
    
    # Host async review
    host_decision = await_host_review(state)           # Chainlit
    if host_decision == "declined":
        state = regenerate_scenario(state)             # Loop back
        host_decision = await_host_review(state)
    
    state.scenario_status = "active"
    show_shared_link(state.room_id)                    # Chainlit
    show_starting_turn(state)

on_student_join(room_id):
    state = load_state(room_id)                        # LangGraph checkpoint
    render_current_turn(state)                         # Chainlit - show choices

on_simulation_turn():
    while state.scenario_status == "active":
        choice = get_student_choice()                  # Chainlit - multiple choice
        state.last_student_choice = choice
        state = process_choice_with_feedback(state)    # LangGraph + PydanticAI
        render_feedback_and_next_turn(state)           # Chainlit

on_scenario_complete():
    debrief = build_debrief(state)                     # PydanticAI (pass/fail)
    render_debrief(debrief)                            # Chainlit
    log_run(state, debrief)                            # MLflow (golden scenarios, traces)
```

### Generation flow pseudocode

```python
run_generation_flow(state):
    state.scenario_draft = generate_complete_scenario(state.host_config)
    # Scenario now contains all turns with multiple choice options

    while state.retry_count < MAX_RETRIES:
        results = run_judges(state.scenario_draft)

        if results.pass_all:
            state.validated_scenario = state.scenario_draft
            state.host_review_status = "pending"
            return state

        state.scenario_draft = refine_scenario(
            draft=state.scenario_draft,
            feedback=results.feedback
        )
        state.retry_count += 1

    # Fallback to golden scenario from MLflow
    state.validated_scenario = load_golden_scenario()
    state.host_review_status = "pending"
    return state
```

### Simulation turn pseudocode

```python
run_simulation_turn(state):
    current_turn = get_current_turn(state)
    
    # Student selects from pre-written choices
    selected_choice = current_turn.choices[student_selection]
    
    # AI generates personalized feedback
    result = generate_feedback(
        scenario=state.validated_scenario,
        current_turn=current_turn,
        selected_choice=selected_choice
    )
    
    state.transcript.append(result.selected_choice)
    state.key_learning_moments.extend(result.learning_moments)
    
    # Advance to next turn based on choice's next_turn_id
    if result.is_complete:
        state.scenario_status = "complete"
        state.completion_status = determine_pass_fail(state.key_learning_moments)
    else:
        state.current_turn_id = selected_choice.next_turn_id
    
    return state
```

## Interfaces

Define clear typed contracts between components.

### Host Config (Input)
```python
HostConfig:
    num_participants   # int (1-20)
    activity_type      # "canyoneering" | "skiing" | "hiking"
    difficulty         # "low" | "med" | "high"
```

### Generator output
```python
ScenarioDraft:
    title
    setting
    patient_summary
    hidden_truth
    learning_objectives
    turns              # list of ScenarioTurn (3-5 turns)
    starting_turn_id
```

### Turn structure
```python
ScenarioTurn:
    turn_id
    narrative_text
    hidden_state       # dict - secret medical info
    scene_state        # dict - visible conditions
    choices            # list of ChoiceOption (2-3 choices)
    is_starting_turn

ChoiceOption:
    choice_id
    description
    is_correct         # bool - medically optimal?
    next_turn_id       # str | null (null = scenario ends)
```

### Simulation feedback output
```python
SimulationResult:
    selected_choice    # ChoiceOption selected
    feedback           # str - AI personalized feedback
    learning_moments   # list[str]
    next_turn          # ScenarioTurn | null
    is_complete        # bool
```

### Judge output
```python
JudgeResult:
    passed
    confidence
    issues
    suggested_fixes
```

### Debrief output
```python
DebriefReport:
    summary
    key_mistakes
    strong_actions
    best_next_actions
    teaching_points
    completion_status  # "pass" | "fail" (learning-focused, not graded)
```

## Development Notes

- Prefer one monolithic Python app for the hackathon.
- Avoid splitting frontend and backend early.
- Use LangGraph checkpointing with DragonflyDB for state persistence (no naked Redis calls).
- Define model assignments in `config/models.yaml` with per-agent overrides and default model.
- Keep prompts simple, but keep schemas strict.
- Build the happy path first: host config → generation → single lead student → debrief.
- Add validation loop and host review after basic flow works.
- Students are anonymous; focus on collaborative learning, not user management.
- Treat MLflow as observability + golden scenario storage; A/B testing is a stretch goal.
- Optimize for a stable live demo over maximum feature count.
- Hybrid approach: Pre-generated scenarios with multiple choice provides reliability while AI feedback preserves interactivity.
