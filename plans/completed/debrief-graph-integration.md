# Plan: Integrate Debrief Agent as Graph Node

## Goal
Move the debrief from a standalone function call to a terminal node in the simulation graph, so the graph owns the complete workflow and the final state includes the debrief report.

## Why
1. The graph should own its own completion -- callers shouldn't need to know about post-processing
2. The debrief result should be in graph state so the checkpointer snapshot is complete
3. MLflow tracing should be unified under the simulation session
4. When Chainlit comes, `graph.ainvoke()` should return everything

## Changes

### 1. `src/summit_sim/graphs/state.py`
- Import `DebriefReport` from `summit_sim.schemas`
- Add `debrief_report: DebriefReport | None` to `StudentState`

### 2. `src/summit_sim/graphs/student.py`
- Import `generate_debrief` from `summit_sim.agents.debrief`
- Add new node function:
  ```python
  async def generate_debrief_node(state: StudentState) -> dict:
      """Generate debrief report after simulation completes."""
      debrief_report = await generate_debrief(
          transcript=state["transcript"],
          scenario_draft=state["scenario_draft"],
          scenario_id=state["scenario_id"],
      )
      return {"debrief_report": debrief_report}
  ```
- Update `check_completion` to route to `"generate_debrief"` instead of `END` when complete:
  ```python
  def check_completion(state: StudentState) -> str:
      if state["is_complete"]:
          return "generate_debrief"
      return "present_turn"
  ```
- Wire the new node in `create_student_graph`:
  ```python
  workflow.add_node("generate_debrief", generate_debrief_node)
  # Update conditional edges
  workflow.add_conditional_edges(
      "update_state",
      check_completion,
      {
          "generate_debrief": "generate_debrief",
          "present_turn": "present_turn",
      },
  )
  workflow.add_edge("generate_debrief", END)
  ```

### 3. `tests/test_simulation_graph.py`
- Add `"debrief_report": None` to all initial state dicts (fixtures and inline)
- In `test_full_three_turn_simulation`:
  - Mock `summit_sim.graphs.simulation.generate_debrief` (the imported function)
  - Have it return a mock `DebriefReport`
  - Assert `state["debrief_report"]` is the mock report in final state
- Update `check_completion` tests: `"__end__"` becomes `"generate_debrief"` for the complete case

### 4. `tests/test_debrief.py`
- No changes needed -- these test the standalone function which still exists

### 5. `notebooks/summit-sim-demo.ipynb`
- Move debrief call inside the `simulation_session` context (it happens automatically now)
- The debrief cell (cell 6) reads `current_state["debrief_report"]` instead of calling `generate_debrief`
- Can optionally log debrief metrics inside the session context too

### 6. `src/summit_sim/agents/debrief.py`
- No changes -- the graph node calls the existing function

## Test impact
- `check_completion` test for complete case: assert `"generate_debrief"` instead of `"__end__"`
- Full cycle test: mock debrief agent, verify `debrief_report` in final state
- All existing debrief unit tests remain valid

## Risk
- Low risk -- the debrief is a terminal linear node, not part of the cycle
- The `generate_debrief` function is already tested independently
- Graph structure change is additive (new node + edge change)
