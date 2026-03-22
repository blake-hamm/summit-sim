# Summit Sim Architecture

## Overview

Summit Sim is an AI-powered wilderness rescue training simulator.

The app has one main flow:
1. Instructor configures a scenario.
2. The system generates and validates the case.
3. The student works through the scenario step by step.
4. The system produces a debrief at the end.

This project uses:
- **Chainlit** for the chat-based frontend and per-user session handling
- **LangGraph** for orchestration, routing, retries, and shared state
- **PydanticAI** for agent calls with structured outputs
- **MLflow** for traces, runs, and lightweight eval logging

## Stack

### Frontend
- **Chainlit**
- Used for:
  - Instructor config UI
  - Student chat interface
  - Status/loading messages
  - End-of-scenario debrief view

### Workflow
- **LangGraph**
- Used for:
  - Scenario generation workflow
  - Validation loop
  - Interactive turn-by-turn simulation
  - Shared scenario state
  - Routing to debrief when the scenario ends

### Agents
- **PydanticAI**
- Used for:
  - Scenario Generator
  - Safety Judge
  - Realism Judge
  - Pedagogy Judge
  - Refiner
  - Simulation Agent
  - Debrief Agent

### Observability
- **MLflow**
- Used for:
  - Trace logging
  - Prompt/version tracking
  - Simple eval metrics
  - Debugging failed generations

## Core Components

### 1. Instructor Config
Collects:
- group size
- difficulty
- environment
- location
- scenario type

**Tool used:** Chainlit

### 2. Scenario Generator
Creates the first draft of the rescue scenario from instructor inputs.

**Tool used:** PydanticAI inside a LangGraph node

### 3. Validation Loop
Runs several judges against the generated scenario:
- safety
- realism
- pedagogy

If the scenario fails, a refiner updates it and the judges run again.

**Tools used:** LangGraph for orchestration, PydanticAI for each judge/refiner

### 4. Simulation Engine
Runs the live scenario once validation passes.

It:
- accepts student input
- updates hidden patient and scene state
- returns the next narrative turn
- introduces complications when appropriate

**Tools used:** LangGraph + PydanticAI

### 5. Debrief
Summarizes:
- what the student did well
- mistakes
- better actions
- teaching points
- score summary

**Tools used:** PydanticAI, optionally MLflow for logged evaluation data

## Shared State

Use one shared graph state object for the whole scenario.

Example fields:

```python
class AppState:
    teacher_config
    scenario_draft
    validated_scenario
    hidden_medical_state
    scene_safety_state
    transcript
    last_student_action
    last_feedback
    score
    penalties
    validation_results
    retry_count
    scenario_status
```

### Notes
- Keep student-visible text separate from hidden simulation state.
- Make all agent outputs structured and typed.
- Add a max retry count for validation.
- Keep one fallback "golden scenario" for demo reliability.

## Flow

### High-level pseudocode

```python
on_start():
    config = get_instructor_config()            # Chainlit
    state.teacher_config = config

    state = run_generation_flow(state)         # LangGraph
    show_intro_turn(state)                     # Chainlit

    while state.scenario_status == "active":
        student_input = get_student_message()  # Chainlit
        state.last_student_action = student_input
        state = run_simulation_turn(state)     # LangGraph + PydanticAI
        render_turn(state)                     # Chainlit

    debrief = build_debrief(state)             # PydanticAI
    render_debrief(debrief)                    # Chainlit
    log_run(state, debrief)                    # MLflow
```

### Generation flow pseudocode

```python
run_generation_flow(state):
    state.scenario_draft = generate_scenario(state.teacher_config)

    while state.retry_count < MAX_RETRIES:
        results = run_judges(state.scenario_draft)

        if results.pass_all:
            state.validated_scenario = state.scenario_draft
            state.scenario_status = "active"
            return state

        state.scenario_draft = refine_scenario(
            draft=state.scenario_draft,
            feedback=results.feedback
        )
        state.retry_count += 1

    state.validated_scenario = load_golden_scenario()
    state.scenario_status = "active"
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
    state.score += turn.score_delta
    state.penalties += turn.penalty_delta

    if turn.is_complete:
        state.scenario_status = "complete"

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
    score_delta
    penalty_delta
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
    final_score
```

## Development Notes

- Prefer one monolithic Python app for the hackathon.
- Avoid splitting frontend and backend early.
- Keep prompts simple, but keep schemas strict.
- Build the happy path first, then add judges and refinement.
- Treat MLflow as observability first, not a big eval platform.
- Optimize for a stable live demo over maximum feature count.
