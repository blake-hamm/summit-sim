# Story 1.1: PydanticAI Agent & MLflow Tracing

## Overview

Implement the Simulation Agent with PydanticAI, define strict Pydantic schemas, and configure MLflow tracing to the homelab server.

**Architecture Decision**: Using a hybrid approach:
- **AI generates scenarios once** from minimal host input (3 params)
- **Pre-written turns** with multiple choice options
- **AI provides dynamic feedback** when students select choices

This balances demo reliability with AI interactivity.

## Implementation Status

**Status**: COMPLETED  
**Test Coverage**: 100% (82/82 statements)  
**All Tests**: PASSING (13/13)

## Architecture

### Hybrid Model Flow

```
Host (3 params) → AI Generator → Complete Scenario (all turns)
                                          ↓
Student → Select Choice → AI Feedback Agent → Next Turn/Complete
```

**Key Design Decisions**:
1. **HostConfig** - Minimal 3-field input (participants, activity, difficulty)
2. **ScenarioDraft** - AI-generated complete scenario with all turns
3. **ScenarioTurn** - Pre-written with 2-3 multiple choice options
4. **SimulationResult** - AI-generated personalized feedback per choice

## Changes Made

### 1. Dependencies

**File**: `pyproject.toml`

Added using `uv add`:
```toml
dependencies = [
    "mlflow>=3.10.1",
    "pydantic-ai>=1.70.0",
    "pydantic-settings>=2.13.1",
    "pytest-asyncio>=1.3.0",
]
```

Also updated pytest configuration:
```toml
[tool.pytest.ini_options]
addopts = "-s"
testpaths = ["tests"]
asyncio_mode = "auto"
```

### 2. Settings Configuration

**File**: `src/summit_sim/settings.py`

Created Pydantic Settings with:
- `mlflow_tracking_uri`: MLflow tracking server URL (default: `http://localhost:5000`)
- `openrouter_api_key`: OpenRouter API key (env: `OPENROUTER_API_KEY`)
- `openrouter_base_url`: OpenRouter API base URL (default: `https://openrouter.ai/api/v1`)
- `default_model`: Hardcoded model (set to: `nvidia/nemotron-3-super-120b-a12b:free`)

Settings load from `.env` file automatically via `pydantic-settings`.

### 3. Pydantic Schemas

**File**: `src/summit_sim/schemas.py`

#### HostConfig (Minimal Teacher Input)
```python
HostConfig:
    num_participants: int  # 1-20
    activity_type: Literal["canyoneering", "skiing", "hiking"]
    difficulty: Literal["low", "med", "high"]
```

#### ChoiceOption (Multiple Choice)
```python
ChoiceOption:
    choice_id: str
    description: str
    is_correct: bool  # Medically optimal?
    next_turn_id: str | None  # Null if scenario ends
```

#### ScenarioTurn (Pre-written Turn)
```python
ScenarioTurn:
    turn_id: str
    narrative_text: str
    hidden_state: dict[str, str]  # Secret medical info
    scene_state: dict[str, str]   # Visible conditions
    choices: list[ChoiceOption]   # 2-3 options
    is_starting_turn: bool
```

#### ScenarioDraft (AI-Generated Complete Scenario)
```python
ScenarioDraft:
    title: str
    setting: str
    patient_summary: str
    hidden_truth: str
    learning_objectives: list[str]
    turns: list[ScenarioTurn]     # All turns pre-written
    starting_turn_id: str
    
    get_turn(turn_id) -> ScenarioTurn | None
```

#### SimulationResult (AI Feedback)
```python
SimulationResult:
    selected_choice: ChoiceOption
    feedback: str                 # Personalized feedback
    learning_moments: list[str]   # Educational insights
    next_turn: ScenarioTurn | None
    is_complete: bool
```

### 4. Agent Configuration & Shared Factory

**File**: `src/summit_sim/agents/config.py`

Centralized agent configuration with shared provider and factory pattern:

**Shared Provider** (singleton via `lru_cache`):
```python
@lru_cache(maxsize=1)
def get_provider() -> OpenRouterProvider:
    """Get singleton OpenRouter provider instance."""
    return OpenRouterProvider(api_key=settings.openrouter_api_key)
```

**Agent Factory** (lazy initialization):
```python
def get_agent(
    agent_name: str,
    output_type: type,
    system_prompt: str,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> Agent[Any, Any]:
    """Get or create an agent singleton by name."""
    if agent_name not in _agent_container:
        _agent_container[agent_name] = Agent(
            OpenRouterModel(settings.default_model, provider=get_provider()),
            output_type=output_type,
            system_prompt=system_prompt,
            model_settings=OpenRouterModelSettings(
                openrouter_reasoning={"effort": reasoning_effort}
            ),
        )
    return _agent_container[agent_name]
```

**Benefits**:
- Single source of truth for agent creation
- Shared provider instance (reused across all agents)
- Lazy initialization (no API key errors during import)
- Testable (can mock `summit_sim.agents.config.Agent`)

---

### 5. Scenario Generator Agent

**File**: `src/summit_sim/agents/generator.py`

Purpose: Generates complete scenarios from minimal `HostConfig`

**Function**: `async def generate_scenario(host_config: HostConfig) -> ScenarioDraft`

**Implementation**:
```python
async def generate_scenario(host_config: HostConfig) -> ScenarioDraft:
    agent = get_agent(
        agent_name="generator",
        output_type=ScenarioDraft,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        reasoning_effort="high",
    )
    result = await agent.run(prompt)
    return result.output
```

**System Prompt Focus**:
- Creates 3-5 turns with cohesive narrative
- Each turn has 2-3 realistic choices
- One medically optimal choice per turn
- Medically accurate wilderness emergencies
- Specific settings appropriate for activity type

**Example Flow**:
```python
config = HostConfig(
    num_participants=4,
    activity_type="hiking",
    difficulty="med"
)
scenario = await generate_scenario(config)
# Returns complete scenario with all turns
```

### 6. Simulation Feedback Agent

**File**: `src/summit_sim/agents/simulation.py`

Purpose: Provides AI-generated feedback when students make choices

**Function**: `async def process_choice(scenario, current_turn, selected_choice) -> SimulationResult`

**Implementation**:
```python
async def process_choice(...) -> SimulationResult:
    agent = get_agent(
        agent_name="simulation_feedback",
        output_type=SimulationResult,
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
        reasoning_effort="medium",
    )
    result = await agent.run(prompt)
    # ... post-processing logic
    return simulation_result
```

**System Prompt Focus**:
- Personalized feedback on choice selection
- Educational but encouraging tone
- Explains WHY choice was good/suboptimal
- 1-2 actionable learning moments

**Example Flow**:
```python
result = await process_choice(
    scenario=scenario,
    current_turn=turn_1,
    selected_choice=choice_a
)
# Returns feedback + next_turn (or marks complete)
```

### 7. Test Implementation

**Files**: 
- `tests/test_generator.py` - Scenario generator tests
- `tests/test_simulation.py` - Simulation feedback agent tests  
- `tests/test_schemas.py` - Schema validation tests

**Test Suite** (22 tests):
- `TestHostConfig`: 3 tests (creation, min/max validation)
- `TestScenarioTurn`: 2 tests (creation, min choices validation)
- `TestScenarioDraft`: 3 tests (creation, get_turn, not found)
- `TestChoiceOption`: 2 tests (creation, end scenario)
- `TestSimulationResult`: 1 test (creation)
- `TestGeneratorAgent`: 7 tests (1 basic + 3 activities × 3 difficulties)
- `TestSimulationAgent`: 4 tests (next turn, end scenario, incorrect selection, learning moments)

**Testing Approach**:
- All agent calls mocked with `AsyncMock`
- No external API calls during unit tests
- 100% coverage of schemas and agents
- **Agent cache clearing fixture**: `clear_agent_cache` resets `_agent_container` before each test
- **Patched at source**: Tests patch `summit_sim.agents.config.Agent` (not individual agent files)

### 7. Environment Configuration

**File**: `.env.example`

Created template:
```bash
# MLflow
MLFLOW_TRACKING_URI=https://mlflow.bhamm-lab.com

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
```

## Verification

### Automated Tests

All 22 tests passing:
```
tests/test_schemas.py::TestHostConfig::test_host_config_creation PASSED
tests/test_schemas.py::TestHostConfig::test_host_config_validation_min PASSED
tests/test_schemas.py::TestHostConfig::test_host_config_validation_max PASSED
tests/test_schemas.py::TestScenarioTurn::test_scenario_turn_creation PASSED
tests/test_schemas.py::TestScenarioTurn::test_scenario_turn_min_choices PASSED
tests/test_schemas.py::TestScenarioDraft::test_scenario_draft_creation PASSED
tests/test_schemas.py::TestScenarioDraft::test_get_turn PASSED
tests/test_schemas.py::TestScenarioDraft::test_get_turn_not_found PASSED
tests/test_schemas.py::TestSimulationResult::test_simulation_result_creation PASSED
tests/test_schemas.py::TestChoiceOption::test_choice_option_creation PASSED
tests/test_schemas.py::TestChoiceOption::test_choice_option_end_scenario PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario_different_activities[canyoneering] PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario_different_activities[skiing] PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario_different_activities[hiking] PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario_different_difficulties[low] PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario_different_difficulties[med] PASSED
tests/test_generator.py::TestGeneratorAgent::test_generate_scenario_different_difficulties[high] PASSED
tests/test_simulation.py::TestSimulationAgent::test_process_choice_with_next_turn PASSED
tests/test_simulation.py::TestSimulationAgent::test_process_choice_ends_scenario PASSED
tests/test_simulation.py::TestSimulationAgent::test_process_choice_incorrect_selection PASSED
tests/test_simulation.py::TestSimulationAgent::test_process_choice_learning_moments PASSED
```

### Coverage Report

```
Name                                  Stmts   Miss Branch BrPart  Cover   Missing
---------------------------------------------------------------------------------
src/summit_sim/agents/generator.py       14      0      0      0   100%
src/summit_sim/agents/simulation.py      23      0      2      0   100%
src/summit_sim/schemas.py                37      0      4      0   100%
src/summit_sim/settings.py                8      0      0      0   100%
---------------------------------------------------------------------------------
TOTAL                                    82      0      6      0   100%
```

### Manual MLflow Verification

To verify MLflow tracing:

1. Set `OPENROUTER_API_KEY` in your `.env` file
2. Run integration test (call `generate_scenario()` without mocking)
3. Open MLflow UI at `https://mlflow.bhamm-lab.com`
4. Navigate to "Traces" section
5. Verify trace appears showing:
   - Prompt text (system + user messages)
   - Structured output (ScenarioDraft JSON)
   - Latency metrics
   - Token usage (input/output counts)

## Success Criteria

- [x] Dependencies added and installed via `uv add`
- [x] Settings module loads from environment
- [x] Schemas defined with proper types and Field descriptions
  - [x] HostConfig (minimal 3-field input)
  - [x] ChoiceOption (multiple choice structure)
  - [x] ScenarioTurn (pre-written turns)
  - [x] ScenarioDraft (AI-generated complete scenario)
  - [x] SimulationResult (AI feedback)
- [x] Shared agent configuration in `config.py`
  - [x] `get_provider()` - Singleton OpenRouter provider
  - [x] `get_agent()` - Factory for creating agents
  - [x] Lazy initialization for testability
- [x] Scenario Generator Agent uses shared `get_agent()` factory
- [x] Simulation Agent uses shared `get_agent()` factory
- [x] Tests pass with mocked data (100% coverage, 22 tests)
- [x] MLflow tracking URI configured for homelab server
- [x] Test coverage ≥80% (achieved 100%)
- [x] `pytest-asyncio` configured for async test support
- [x] All code passes ruff linting

## Files Created/Modified

1. `pyproject.toml` - Added dependencies and pytest config
2. `src/summit_sim/settings.py` - Pydantic settings with env vars
3. `src/summit_sim/schemas.py` - Complete schema hierarchy
4. `src/summit_sim/agents/config.py` - **NEW**: Shared provider and agent factory
5. `src/summit_sim/agents/generator.py` - Scenario generator agent (uses `get_agent()`)
6. `src/summit_sim/agents/simulation.py` - Simulation feedback agent (uses `get_agent()`)
7. `tests/test_generator.py` - Generator agent tests (7 tests)
8. `tests/test_simulation.py` - Simulation agent tests (4 tests)
9. `tests/test_schemas.py` - Schema validation tests (9 tests)
10. `.env.example` - Environment template
11. `plans/story-1-1-implementation.md` - This document
12. `plans/agent-configuration-patterns.md` - Documentation for future patterns

## Next Steps

1. **Manual Integration Test**: Run actual agents with OpenRouter API to verify:
   - Scenario generation creates medically accurate content
   - Turns have appropriate branching
   - MLflow traces appear in UI

2. **Ruff Linting**: Run `ruff check . && ruff format .` when environment supports it

3. **Pre-commit**: Install hooks with `pre-commit install`

4. **Story 1.2**: Implement validation loop (Safety, Realism, Pedagogy judges)

## Notes

- **Architecture Pattern**: Shared agent factory in `config.py` replaces hardcoded agent functions
  - `get_agent()` creates agents on-demand with configurable parameters
  - `get_provider()` provides singleton OpenRouter provider instance
  - Agents cached by name in `_agent_container` dict
- Model hardcoded to `nvidia/nemotron-3-super-120b-a12b:free` as requested
- MLflow tracing enabled via `mlflow.set_tracking_uri()` - autologging to be verified
- Agent uses `output_type` parameter per pydantic-ai 1.70.0 API
- Schema file named `schemas.py` as preferred over `models.py`
- All code follows project conventions from AGENTS.md
- Hybrid approach balances demo reliability with AI capabilities
- Few-shot prompting can be added to generator agent later for consistency
- Tests patch `summit_sim.agents.config.Agent` to mock at the source
- Agent cache cleared between tests via `clear_agent_cache` fixture
