# SFT_RL_Circuits

## Setup
- Python version: 3.10 (see `.python-version`).
- Install developer tooling: `python -m pip install -r requirements/dev.txt` (includes Black, Ruff, Pytest).
- Package metadata and build config live in `pyproject.toml`; source code is under `src/sft_rl_circuits`.

## CI
GitHub Actions workflow runs format check (`black`), linting (`ruff`), and tests (`pytest`) on pushes and pull requests via `.github/workflows/ci.yml`.
