# Agent Instructions for summit-sim

Summit-Sim is an AI wilderness rescue simulator using multi-agent validation to generate medically safe, interactive backcountry emergencies for first-responder training.

> **CRITICAL: This repo runs on NixOS and requires Nix. Run `nix develop` BEFORE any commands.**
>
> **Hackathon Context**: 2-week sprint. Prioritize working code over perfection, but maintain hygiene (types, tests, docs). Use existing patterns -- don't over-engineer.
>
> **Core Flow**: Host configures scenario -> AI generates scenario -> Validation judges check it -> Host reviews -> Students join via link -> Simulation runs -> Debrief at end. See `plans/high-level-arch.md` for full details.

## Hard Boundaries

These rules are absolute. Do not violate them under any circumstances.

- **No git commands.** Do not run `git add`, `git commit`, `git push`, or any other version control commands. The human developer handles all git operations. Your job is only to write, test, and verify code.
- **No bypassing the agent factory.** All agents must be created via `get_agent()` from `agents/config.py`. Never instantiate `Agent(...)` directly in application code.
- **Use Nix-provided ruff only.** The `.venv` installs a dynamically-linked `ruff` binary that does NOT work on NixOS. Always use the `ruff` binary from the Nix dev shell. Do not rely on `.venv/bin/ruff` or run pre-commit hooks for linting -- run ruff and coverage checks manually instead.
- **No external API calls in tests.** All LLM calls must be mocked via `unittest.mock.AsyncMock`. Patch at `summit_sim.agents.config.Agent`, not at import sites.

## Environment Setup

### Option 1: Nix (Recommended for development)

```bash
nix develop                     # REQUIRED -- enters dev shell with Python 3.12, ruff, uv, pre-commit
uv sync --all-extras            # One-time: install all dependencies
source .venv/bin/activate       # Activate virtual environment
```

### Option 2: Docker Compose (Alternative)

Docker Compose provides an isolated environment with explicit configuration, useful for:
- Testing deployment-like setups
- Avoiding environment variable issues (`.env` loading, AWS credentials)
- Hot reloading during development

```bash
# Build and run with hot reload
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Key differences from Nix:**
- Environment variables loaded from `.env` file via docker-compose.yml
- No access to `~/.aws/credentials` (prevents AWS credential issues)
- Automatic hot reload on file changes (via docker-compose.override.yml)
- Runs on port 8000

## Quality Gates (run after every change)

Every change must pass these two checks before declaring a task complete:

```bash
nix develop -c 'ruff check --fix && ruff format'   # Lint + format (use Nix-provided ruff)
nix develop -c 'coverage run && coverage report'  # Tests + coverage (must be >=80%)
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

### Tech Stack

| Framework | Status | Role |
|-----------|--------|------|
| PydanticAI | Active | Agent calls, structured outputs, OpenRouter provider |
| LangGraph | Active | Workflow orchestration, state, interrupt/resume |
| MLflow | Active | Tracing, autologging, experiment tracking |
| Pydantic + pydantic-settings | Active | Schemas, env config |
| OpenRouter | Active | LLM provider (default: gemini-3.1-flash-lite-preview) |
| Chainlit | Active | Web UI framework (teacher/student flows) |
| DragonflyDB | Not yet implemented | Persistence (currently using InMemorySaver) |

## Development Tips

### Adding a New Agent
1. Define the output schema in `schemas.py`
2. Create the agent module in `agents/` using `get_agent()` from `config.py`
3. Write system prompt in the module, register user prompt template in MLflow as `prompts:/{AGENT_NAME}-user@latest`
4. Post-process deterministic fields outside the LLM (see `agents/simulation.py` for example)
5. Test with mocked LLM calls; verify with a notebook for live E2E

## Architecture Decisions
- **Prioritize features**: Get new functionality working first, polish later
- **Keep it simple**: Avoid premature abstraction and security overthinking
- **Read the plan first**: Check `plans/` for context before implementation
