# Contributing

## Setup

```bash
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
uv pip install -e ".[dev]"            # add ,craftax for the real game: ".[dev,craftax]"
pre-commit install
```

## Before every commit

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format src tests
uv run --no-sync mypy
uv run --no-sync pytest               # fails under 80% coverage
uv run --no-sync pytest -m craftax --no-cov   # with the craftax extra installed
```

CI runs the same checks on Python 3.10 and 3.12.

## Rules

- Package code in `src/jepa_game_agent`, one responsibility per module;
  experiments (Colab notebooks, scripts) in `experiments/` and only call the
  package.
- Every new function or class gets a unit test. Tests run on a CPU in
  seconds: use `GridWorld` and small configs; mark slow or GPU-only tests.
- Configs are frozen dataclasses validated on creation; no magic numbers in
  code paths.
- Results: JSON with the config and the git commit; report failures too.
- Commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Specs first: a new part is described in `docs/SPECS.md` before it is built.
