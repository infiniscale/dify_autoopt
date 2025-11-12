# Repository Guidelines

## Project Structure & Module Organization
- Source lives under `src/` and is organized by capability: `auth/`, `workflow/`, `config/`, `executor/`, `collector/`, `report/`, `optimizer/`, `utils/`. Each folder currently contains a `README.md` describing intended modules.
- The entry point referenced in README is `src/main.py`. If missing, add it and keep imports under `src/` (use package-style imports).
- Place test code in `tests/`, mirroring `src/` (e.g., `tests/optimizer/test_optimization_engine.py`). Keep large assets out of the repo; use `assets/` or external storage if needed.

## Build, Test, and Development Commands
- Environment: Python 3.8+ with a virtualenv.
  - Create venv: `python -m venv .venv && source .venv/bin/activate`
  - Install deps (when added): `pip install -r requirements.txt` or `pip install -e .` (pyproject).
- Run entry point (per README modes): `python src/main.py --mode test|optimize|report`.
- Run tests (pytest): `python -m pytest -q` and coverage: `pytest --cov=src --cov-report=term-missing`.

## Coding Style & Naming Conventions
- Follow PEP 8; 4-space indentation; max line length 100–120.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Use type hints and module docstrings. Prefer dependency injection over globals.
- Formatting/linting (recommended): `black` and `ruff`. Example: `ruff check src tests` and `black src tests`.

## Testing Guidelines
- Framework: `pytest`. Put unit tests under `tests/` with files named `test_*.py` and mirror package layout.
- Target coverage: 80%+ for changed lines. Include edge cases and error paths. Use fixtures for external services.
- For I/O or network, isolate with fakes; do not hit real Dify endpoints in unit tests.

## Commit & Pull Request Guidelines
- Flow: Git Flow. Branch from `develop` using `feature/<scope>`, `fix/<scope>`, or `hotfix/<scope>`. Hooks under `.claude/hooks/` validate branch/push.
- Commits: Imperative mood, concise summary, optional body explaining why (e.g., `feat(optimizer): add version manager`).
- PRs: Target `develop`. Include purpose, linked issues, test plan, and screenshots for reports when relevant. Keep PRs focused and small.
- CI/Automation: GitHub Actions in `.github/workflows` run Claude-based review/assistance; never commit secrets—use repo/environment secrets.

## Security & Configuration Tips
- Do not commit credentials or API keys. Prefer environment variables and `.env` (excluded) and GitHub Secrets.
- Store configuration in YAML under `src/config/` (or `config/` if promoted), and validate before runtime.
