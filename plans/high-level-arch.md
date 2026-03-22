# Summit Sim Architecture

## Overview

Summit Sim is an AI-powered wilderness rescue training simulator for collaborative learning.

The app has one main flow:
1. Host configures a scenario and triggers generation.
2. The system validates for medical accuracy and realism.
3. Host reviews the scenario asynchronously (accept/decline).
4. Students join via shared link; one lead student drives actions, others observe.
5. The lead student works through the scenario step by step.
6. The system produces a learning-focused debrief (pass/fail, not graded mastery).

This project uses:
- **Chainlit** for the chat-based frontend and session handling
- **LangGraph** for orchestration, routing, retries, and shared state
- **DragonflyDB** (Redis-compatible) for persistence via LangGraph checkpointing
- **PydanticAI** for agent calls with structured outputs
- **OpenRouter** for multi-model LLM access with per-agent configuration
- **MLflow** for traces, golden scenarios, A/B testing, and eval logging
- **YAML config** for model assignments (per-agent and default)

## Stack

### Frontend
- **Chainlit**
- Used for:
  - Host configuration UI
  - Host async review interface (accept/decline scenarios)
  - Student chat interface with shared room links
  - Status/loading messages
  - End-of-scenario debrief view

### Workflow & State
- **LangGraph** with **DragonflyDB** persistence
- Used for:
  - Scenario generation workflow with checkpointing
  - Validation loop with retry logic
  - Interactive turn-by-turn simulation
  - Shared scenario state across users
  - Routing to debrief when the scenario ends

### Agents
- **PydanticAI** via **OpenRouter**
- Used for:
  - Scenario Generator
  - Safety Judge (medical accuracy validation)
  - Realism Judge
  - Pedagogy Judge
  - Refiner
  - Simulation Agent
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
Collects scenario parameters:
- group size
- difficulty
- environment
- location
- scenario type

Generates shareable room link for students to join.

**Tool used:** Chainlit

### 2. Scenario Generator
Creates the first draft of the rescue scenario from host inputs.

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
- accepts lead student input
- updates hidden patient and scene state
- returns the next narrative turn
- introduces complications when appropriate

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
    host_config                      # Scenario parameters
    scenario_draft
    validated_scenario
    host_review_status               # "pending" | "accepted" | "declined"
    hidden_medical_state
    scene_safety_state
    transcript                       # List of turn records
    last_student_action
    validation_results
    retry_count
    scenario_status                  # "generating" | "review_pending" | "active" | "complete"
    completion_status                # "incomplete" | "pass" | "fail" (learning-focused)
    key_learning_moments
    model_config                     # Which models were used (for A/B testing)
```

### Notes
- Keep student-visible text separate from hidden simulation state.
- Make all agent outputs structured and typed.
- Add a max retry count for validation before falling back to golden scenario.
- Store golden scenarios in MLflow for demo reliability and few-shot prompting.

## Flow

### High-level pseudocode

```python
on_host_start():
    config = get_host_config()                          # Chainlit
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
    show_intro_turn(state)

on_student_join(room_id):
    state = load_state(room_id)                        # LangGraph checkpoint
    render_current_state(state)                        # Chainlit

on_simulation_turn():
    while state.scenario_status == "active":
        student_input = get_lead_student_message()     # Chainlit
        state.last_student_action = student_input
        state = run_simulation_turn(state)             # LangGraph + PydanticAI
        render_turn(state)                             # Chainlit

on_scenario_complete():
    debrief = build_debrief(state)                     # PydanticAI (pass/fail)
    render_debrief(debrief)                            # Chainlit
    log_run(state, debrief)                            # MLflow (golden scenarios, traces)

### Generation flow pseudocode

```python
run_generation_flow(state):
    state.scenario_draft = generate_scenario(state.host_config)

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
    turn = simulate_next_step(
        scenario=state.validated_scenario,
        transcript=state.transcript,
        hidden_state=state.hidden_medical_state,
        student_action=state.last_student_action
    )

    state.transcript.append(turn.narrative_text)
    state.hidden_medical_state = turn.updated_hidden_state
    state.scene_safety_state = turn.updated_scene_state
    state.key_learning_moments.extend(turn.learning_moments)

    if turn.is_complete:
        state.scenario_status = "complete"
        state.completion_status = determine_pass_fail(state.key_learning_moments)

    return state
```

## Interfaces

Define clear typed contracts between components.

### Generator output
```python
ScenarioDraft:
    title
    setting
    patient_summary
    hidden_truth
    starting_conditions
    learning_objectives
```

### Judge output
```python
JudgeResult:
    passed
    confidence
    issues
    suggested_fixes
```

### Simulation output
```python
SimulationTurn:
    narrative_text
    feedback_on_last_action
    updated_hidden_state
    updated_scene_state
    available_actions
    learning_moments
    is_complete
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
