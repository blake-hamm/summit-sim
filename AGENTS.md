# Agent Instructions for summit-sim

Summit-Sim is an AI wilderness rescue simulator using multi-agent validation to generate medically safe, interactive backcountry emergencies for first-responder training.

## Build/Lint/Test Commands

```bash
# Development environment setup (uses Nix + uv)
nix develop                          # Enter dev shell with Python 3.12, uv, ruff
uv sync --all-extras                # Install all dependencies including dev
source .venv/bin/activate           # Activate virtual environment

# Testing
pytest                              # Run all tests
pytest tests/test_specific.py       # Run single test file
pytest tests/test_specific.py::test_function  # Run single test
pytest -k "test_name"               # Run tests matching pattern
pytest -s                           # Run with print statements (already default)
pytest --cov=src --cov=tests        # Run with coverage

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
