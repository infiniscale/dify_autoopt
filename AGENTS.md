# Repository Guidelines

## Project Structure & Module Organization
- Source lives under `src/` by capability: `auth/`, `workflow/`, `config/`, `executor/`, `collector/`, `report/`, `optimizer/`, `utils/`. Each folder has a `README.md` describing intent.
- Entry point: `src/main.py` (modes: `--mode test|optimize|report`). Keep imports package-style under `src/`.
- Tests: place unit tests in `src/test/` mirroring `src/` (e.g., `src/test/test_logger_basic.py`). Use `assets/` for large non-code files when needed.

## Build, Test, and Development Commands
- Python 3.8+ with a virtualenv.
  - Create venv: `python -m venv .venv && source .venv/bin/activate`
  - Install deps: `pip install -r requirements.txt`
- Run: `python src/main.py --mode test` (or `optimize`, `report`).
- Tests: `python -m pytest -q`
- Coverage: `pytest --cov=src --cov-report=term-missing`

## Coding Style & Naming Conventions
- PEP 8; 4-space indent; line length ≤ 100–120.
- Names: `snake_case` (files/functions), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Use type hints and module docstrings; prefer dependency injection over globals.
- Formatting/linting: `black` and `ruff`.
  - Examples: `ruff check src src/test` and `black src src/test`.

## Testing Guidelines
- Framework: `pytest`; tests named `test_*.py` under `src/test/`.
- Aim for 80%+ coverage of changed code; include edge cases and error paths.
- Isolate I/O or network with fakes; do not hit real Dify endpoints in unit tests.
- Logger example: see `src/test/test_logger_basic.py` for initialization and file sink checks.

## Commit & Pull Request Guidelines
- Flow: Git Flow. Branch from `develop`: `feature/<scope>`, `fix/<scope>`, `hotfix/<scope>`; hooks in `.claude/hooks/` validate branch/push.
- Commits: imperative mood and concise (e.g., `feat(optimizer): add version manager`).
- PRs: target `develop`; include purpose, linked issues, test plan, and relevant screenshots (for reports). Keep PRs focused and small.
- CI: GitHub Actions in `.github/workflows/`; never commit secrets—use repo/environment secrets.

## Security & Configuration Tips
- Never commit credentials or API keys. Use environment variables and `.env` (git-ignored).
- Store validated YAML config under `src/config/` (e.g., `config/logging_config.yaml`).
