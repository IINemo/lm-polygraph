# Repository Guidelines

## Project Structure & Module Organization
- `src/lm_polygraph/`: core Python package (estimators, metrics, utils).
- `scripts/`: CLI entry points (`polygraph_eval`, `polygraph_eval_ood`, `polygraph_server`, `polygraph_normalize`).
- `test/`: pytest suite and Hydra configs (`test/configs`), local fixtures in `test/local`.
- `docs/`: Sphinx documentation.
- `examples/`, `notebooks/`: usage examples and exploratory workflows.
- `dataset_builders/`, `build/`, `workdir/`: data helpers, build artifacts, and experiment outputs.

## Build, Test, and Development Commands
- Create env + install (editable): `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Run tests: `pytest -q`
- Quick eval (example): `HYDRA_CONFIG=examples/configs/polygraph_eval_coqa.yaml polygraph_eval save_path=./workdir/output`
- Offline datasets cache: `HF_DATASETS_OFFLINE=1` (respects `args.cache_path`).

## Coding Style & Naming Conventions
- Follow PEP 8, 4‑space indentation, 88–100 char soft limit.
- Naming: `snake_case` for modules/functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer type hints and concise docstrings for public APIs.
- No enforced linter in repo; keep formatting consistent (use `black`/`ruff` locally if desired).

## Testing Guidelines
- Framework: pytest. Place tests under `test/` with `test_*.py` naming.
- Keep tests deterministic; use small configs under `test/configs` and `workdir/output/test` for outputs.
- Run: `pytest -q`. Aim to cover core logic and error paths; add unit tests when touching `src/` or `scripts/`.

## Commit & Pull Request Guidelines
- Commits: imperative, present tense, concise (e.g., "Add mbr decoding", "Fix eval config path"). Reference issues when relevant.
- PRs: include what/why, key changes, how to reproduce (commands/config), and test evidence. Update docs/examples when behavior changes.
- Ensure tests pass locally before opening PR.

## Security & Configuration Tips
- Do not commit secrets. Configure via env vars: `OPENAI_API_KEY`, `OPENAI_KEY`, `HUGGINGFACE_API_KEY`, `HYDRA_CONFIG`, `HF_DATASETS_OFFLINE`.
- Prefer local caches for HF datasets/models; pin seeds via Hydra (`seed=[...]`).

## Agent-Specific Instructions
- Keep patches minimal and scoped; avoid broad refactors.
- Align changes in `src/` with corresponding tests in `test/` and update docs when needed.
- Use fast search (`rg`) and adhere to this guide for structure and style.

