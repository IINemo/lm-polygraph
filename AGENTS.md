# Repository Guidelines
#
## Project Structure & Module Organization
- Source code is under `src/lm_polygraph/` (core logic, calculators, estimators, model wrappers).
- Scripts for running benchmarks and entry points: `scripts/` (e.g., `polygraph_eval`).
- YAML/Hydra configs: `examples/configs/` and subfolders.
- Tests are located in `tests/`.
- Documentation: `docs/`.

## Build, Test, and Development Commands
- Install dependencies: `pip install -e .`
- Run benchmarks locally: `python scripts/polygraph_eval ...` (see configs in `examples/`)
- Format code: `black .`
- Lint code: `flake8 --extend-ignore E501,F405,F403,E203 --per-file-ignores __init__.py:F401,builder_stat_calculator_simple.py:F401 .`

## Coding Style & Naming Conventions
- Follow [PEP 8](https://pep8.org/) Python standards; use 4 spaces for indentation.
- Format all code with `black` before committing.
- Use descriptive, lower_snake_case for variables/functions; PascalCase for class names.

## Testing Guidelines
- Unit and integration tests use `pytest`. Place new tests in `tests/`, mirroring source structure.
- Name test files as `test_*.py` and functions as `test_*`.
- Run all tests: `pytest` from the repository root.

## Commit & Pull Request Guidelines
- Write clear, concise commit messages (imperative tense, e.g., "Add UE metrics for new estimator").
- Each pull request should summarize changes, link related issues, and describe testing performed.
- Ensure code passes linters and all tests before opening or updating a PR.

## Additional Tips
- Reference the onboarding guide for details on module roles and workflow (`README.md` and `docs/`).
- Use Hydra configs for flexible experiment setup — see examples in `examples/configs/`.
