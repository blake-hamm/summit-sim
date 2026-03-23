# Agent Instructions for summit-sim

Summit-Sim is an AI wilderness rescue simulator using multi-agent validation to generate medically safe, interactive backcountry emergencies for first-responder training.

> **CRITICAL: This repo requires Nix. Run `nix develop` BEFORE any commands or they will fail.**
>
> **Hackathon Context**: This is a 2-week sprint. Prioritize working code over perfection, but maintain basic hygiene (types, tests, docs) to keep velocity high. Use the existing patterns - don't over-engineer.
> 
> **Core Flow**: Host configures scenario → AI generates scenario → Validation judges check it → Host reviews → Students join via link → Simulation runs → Debrief at end. See `plans/high-level-arch.md` for full details.

## Build/Lint/Test Commands

**ALL commands require `nix develop` first. The environment won't work without it.**

```bash
# REQUIRED FIRST: Enter Nix development shell
# Without this step, all commands below will fail
nix develop                          # Enter dev shell with Python 3.12, uv, ruff, pre-commit

# One-time setup (after nix develop)
uv sync --all-extras                # Install all dependencies including dev
source .venv/bin/activate           # Activate virtual environment

# Testing - Essential for rapid iteration
pytest                              # Run all tests (quick feedback)
pytest tests/test_specific.py       # Run single test file (faster)
pytest tests/test_specific.py::test_function  # Run single test (fastest)
pytest -k "test_name"               # Run tests matching pattern
pytest -s                           # Run with print statements enabled
pytest --cov=src --cov=tests        # Run with coverage
pytest -x                           # Stop on first failure (fast feedback)
pytest --tb=short                   # Shorter tracebacks

# Linting and Formatting (must pass before commit)
ruff check .                        # Check all files
ruff check --fix .                  # Check and auto-fix issues
ruff format .                       # Format all files
ruff format --check .               # Check formatting without changes

# Coverage
coverage run -m pytest              # Run tests with coverage
coverage report                     # Show coverage report
coverage html                       # Generate HTML coverage report

# Pre-commit (runs automatically on commit)
pre-commit install                  # Install hooks
pre-commit run --all-files          # Run all hooks manually
```

## Code Style Guidelines

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Default ruff (88 characters)
- **Formatter**: ruff format (not black)
- Format on save enabled in VSCode

### Imports
- Use absolute imports within the package
- Ruff enforces import sorting (I rule)
- Group imports: stdlib, third-party, local
- Use `__init__.py` files to expose public APIs

### Type Hints
- Python 3.12+ required - use modern syntax
- Use `|` for union types (e.g., `str | None`)
- Annotate all function parameters and return types
- Ruff ANN rules enforce type annotations

### Naming Conventions
- **Modules**: lowercase with underscores (snake_case)
- **Classes**: PascalCase
- **Functions/Methods**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Private**: prefix with underscore

### Documentation
- All public functions must have docstrings (D rules)
- Use Google-style or NumPy-style docstrings
- Module-level docstrings encouraged
- Ignored: D203 (1 blank line before class), D213 (multi-line summary second line)

### Error Handling
- Use specific exceptions, avoid bare `except:`
- Use `raise from` when re-raising exceptions
- Prefer `pathlib` over `os.path` (PTH rules)

### Ruff Lint Rules Enabled
- **I**: Import sorting
- **F**: Pyflakes
- **E,W**: pycodestyle errors/warnings
- **N**: pep8-naming
- **D**: pydocstyle
- **ANN**: flake8-annotations
- **B**: flake8-bugbear
- **A**: flake8-builtins
- **T20**: flake8-print
- **PYI**: flake8-pyi
- **Q**: flake8-quotes
- **RET**: flake8-return
- **ARG**: flake8-unused-arguments
- **PTH**: flake8-use-pathlib
- **PL**: Pylint
- **PT**: flake8-pytest-style

## Project Structure

```
src/
  summit_sim/          # Main package
    __init__.py        # Package initialization
tests/                 # Test files
  __init__.py
  test_*.py            # Test modules
notebooks/             # Jupyter notebooks
pyproject.toml         # Project config, dependencies
.flake.nix            # Nix development environment
.pre-commit-config.yaml # Pre-commit hooks
```

## Testing Requirements

- Minimum 80% code coverage enforced
- Use pytest fixtures for test setup
- Test files must be named `test_*.py`
- Test functions must be named `test_*`
- Coverage includes: src/, tests/, notebooks/

## Git Workflow

- Pre-commit hooks run automatically
- Hooks include: ruff lint, ruff format, coverage run, coverage report
- Coverage must be ≥80% for pre-commit to pass
- Use conventional commits if possible

## Rapid Development Tips

### When Adding Features
1. Prefer TDD when it helps clarify requirements, but don't force it - use what works for the feature
2. Focus on happy path first, verify core functionality works before edge cases
3. Use type hints from the start - catches bugs immediately
4. Add docstrings as you go (ruff will remind you)
5. Run `pytest -x` frequently during development

### Debugging
- Use `pytest -s` to see print statements
- Add `breakpoint()` for interactive debugging
- Use `pytest --tb=short` for cleaner error output
- Run single tests with `pytest path/to/test.py::test_func`

### Architecture Decisions
- **Prioritize features**: Get new functionality working first, polish later
- **Happy path focus**: Build testable, verifiable features without over-engineering
- **Build in order**: Host config → generation → single lead student → debrief, THEN add validation loop
- **Keep it simple**: Avoid premature abstraction and security overthinking
- **Monolithic app**: One Python app, no frontend/backend split
- **Use existing patterns**: Follow conventions from similar files
- **One module per domain**: Group related functionality
- **Agent-based design**: Embrace Chainlit + LangGraph + PydanticAI architecture
- **Anonymous users**: No user management, focus on collaborative learning
