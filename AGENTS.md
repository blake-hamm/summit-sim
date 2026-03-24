# Agent Instructions for summit-sim

Summit-Sim is an AI wilderness rescue simulator using multi-agent validation to generate medically safe, interactive backcountry emergencies for first-responder training.

> **CRITICAL: This repo runs on NixOS and requires Nix. Run `nix develop` BEFORE any commands.**
>
> **Hackathon Context**: 2-week sprint. Prioritize working code over perfection, but maintain hygiene (types, tests, docs). Use existing patterns -- don't over-engineer.
>
> **Core Flow**: Host configures scenario -> AI generates scenario -> Validation judges check it -> Host reviews -> Students join via link -> Simulation runs -> Debrief at end. See `plans/high-level-arch.md` for full details.

## Environment Setup

```bash
nix develop                     # REQUIRED -- enters dev shell with Python 3.12, ruff, uv, pre-commit
uv sync --all-extras            # One-time: install all dependencies
source .venv/bin/activate       # Activate virtual environment
```

**NixOS caveat**: The venv installs a dynamically-linked `ruff` binary that does NOT work on NixOS. Always use ruff from the Nix shell (call `ruff` before activating `.venv`, or use the full Nix path). The `pre-commit` hooks for ruff currently fail for this reason -- run ruff and coverage checks manually instead.

## Quality Gates (run after every change)

Every change must pass these two checks before committing:

```bash
ruff check --fix . && ruff format .     # Lint + format (use Nix-provided ruff)
coverage run -m pytest && coverage report  # Tests + coverage (must be >=80%)
```

That's it. If those pass, you're good. Ruff config is in `pyproject.toml` -- let ruff teach you the rules rather than memorizing them.

## Code Style (the essentials)

- **Python 3.12+** -- use modern syntax (`str | None`, not `Optional[str]`)
- **Type-annotate** all function parameters and return types
- **Docstrings** on all public functions -- short summary only, NO `Args:`, `Returns:`, `Raises:` sections (type hints and names should be self-documenting)
- **Absolute imports** within the package
- **pathlib** over `os.path`
- **Specific exceptions** -- no bare `except:`; use `raise from` when re-raising
- 4 spaces, 88-char lines, ruff handles the rest

## Architecture Overview

```
src/summit_sim/
  settings.py           # pydantic-settings, loads from .env
  schemas.py            # All Pydantic models (HostConfig, ScenarioDraft, SimulationResult, etc.)
  tracing.py            # MLflow tracing + session management
  agents/
    config.py           # Shared agent factory: get_agent() -> cached PydanticAI Agent
    generator.py        # Scenario generation (PydanticAI + OpenRouter)
    simulation.py       # Per-turn feedback agent
  graphs/
    state.py            # LangGraph TypedDict states (SimulationState, TranscriptEntry)
    simulation.py       # 5-node LangGraph workflow with interrupt() for human-in-the-loop
tests/
  test_schemas.py       # Schema validation
  test_generator.py     # Generator agent (mocked)
  test_simulation.py    # Simulation agent (mocked)
  test_simulation_graph.py  # Graph nodes + full integration test
notebooks/
  story-1-1-integration-test.ipynb  # E2E with live API
  story-1-2-simulation-graph.ipynb  # Full simulation with MLflow
plans/                  # Story plans -- read before implementing
```

### Key Patterns

- **Agent factory** (`agents/config.py`): `get_agent(name, output_type, system_prompt)` returns a cached PydanticAI `Agent` singleton. Use this for all new agents.
- **Structured outputs**: Agents return Pydantic models directly via PydanticAI's `output_type`.
- **LangGraph state**: Uses `TypedDict` (not Pydantic BaseModel) with `append_reducer` for list fields.
- **Human-in-the-loop**: `interrupt()` in graph nodes, resumed via `Command(resume=value)`.
- **MLflow tracing**: `simulation_session()` context manager wraps graph execution with parent runs.
- **Test isolation**: All LLM calls are mocked via `unittest.mock.AsyncMock`. Patch at `summit_sim.agents.config.Agent`, not at import sites. No external API calls in tests.

### Tech Stack (actual usage)

| Framework | Status | Role |
|-----------|--------|------|
| PydanticAI | Active | Agent calls, structured outputs, OpenRouter provider |
| LangGraph | Active | Workflow orchestration, state, interrupt/resume |
| MLflow | Active | Tracing, autologging, experiment tracking |
| Pydantic + pydantic-settings | Active | Schemas, env config |
| OpenRouter | Active | LLM provider (default: gemini-3.1-flash-lite-preview) |
| Chainlit | Planned | Frontend -- not yet implemented |
| DragonflyDB | Planned | Persistence -- currently using InMemorySaver |

## Current Progress

### Completed
- **Story 1.1**: Schemas, generator agent, simulation feedback agent, MLflow tracing, agent factory (22 tests)
- **Story 1.2**: LangGraph simulation graph, 5-node workflow, interrupt(), transcript tracking, MLflow sessions (12 more tests, 34 total)
- **Story 1.3 pre-reqs**: SimulationState renamed, was_correct added, class_id/scenario_id in state, tracing updated

### Next Up
- **Story 1.3** (`plans/story-1-3-student-debrief.md`): Debrief agent, DebriefReport schema, MLflow debrief metrics. Most pre-reqs already done -- read the plan before starting.

### Backlog
- Validation judges (Safety, Realism, Pedagogy) + Refiner agent
- Chainlit frontend
- DragonflyDB persistence
- YAML-based model config (`plans/backlog/agent-configuration-patterns.md`)
- Generator debrief (`plans/backlog/story-1-4-generator-debrief.md`)

## Development Tips

### Adding a New Agent
1. Define the output schema in `schemas.py`
2. Create the agent module in `agents/` using `get_agent()` from `config.py`
3. Write a clear system prompt; use a user prompt template for variable data
4. Post-process deterministic fields outside the LLM (see `agents/simulation.py` for example)
5. Test with mocked LLM calls; verify with a notebook for live E2E

### Adding a New Graph Node
1. Define any new state fields in `graphs/state.py` (use `append_reducer` for lists)
2. Add the node function in the appropriate graph module
3. Wire it into the graph builder in `create_simulation_graph()`
4. Test the node in isolation, then test the full graph flow

### General Workflow
- Read the relevant plan in `plans/` before starting a story
- Focus on happy path first, then edge cases
- Use `pytest -x` for fast feedback during development
- Run the quality gates before committing
- Notebooks are for live E2E verification, not for production code
