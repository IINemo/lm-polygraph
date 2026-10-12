# Creating a Pull Request

1. Fork the repository
2. Clone the repository to your local machine
3. Create a new branch for your changes
4. Make your changes
5. If at all possible, add tests for your changes. Tests are required for new methods of uncertainty estimation.
6. **Make sure to update documentation if needed**
7. Use flake8 and black to conform to the code style. Github actions will check this automatically using the following command:
    ```bash
    flake8 --extend-ignore E501,F405,F403,E203 --per-file-ignores __init__.py:F401 .
    ```
    Use this command locally to check if it will pass when PR is opened. Install flake8-black beforehand to check for black formatting as well.
8. Run tests with `pytest` and make sure they pass
9. Commit and push your changes
10. Create a pull request to the main branch of the original repository

## Fast offline tests

The default `python -m pytest` command collects only `test/unit`. These tests use
small in-memory inputs and block network sockets. They do not need model weights,
CUDA, API keys, or the full model dependency stack. In a fresh virtual environment:

```bash
python -m pip install -r test/requirements-unit.txt
python -m pip install --no-deps -e .
python -m pytest --disable-socket
```

Every pull request runs this suite on Python 3.10, 3.11, and 3.12 (the versions
listed in `pyproject.toml`), with Hugging Face offline mode enabled. Lint runs in
its own job. Put new deterministic tests in `test/unit`; do not load models or
call external services there.

## Integration tests

Integration suites are selected with explicit paths so their imports and fixtures
never run during default unit test collection. Install the full package in a
separate environment first (`python -m pip install '.[comet]'`). Model tests
download weights and datasets and may take substantial time and disk space:

```bash
python -m pytest -m model test/test_lm_polygraph.py test/test_estimators.py test/test_high_level_api.py
python -m pytest -m api test/local/blackbox
# Requires a CUDA GPU and an additional `pip install vllm`:
python -m pytest -m vllm test/local/test_vllm.py
# Optional, expensive benchmark and vision suites:
python -m pytest -m model test/local/test_benchmark.py test/local/test_estimators_visual.py
```

The API suite uses a local mock OpenAI server; no real API key is required. Its
end-to-end pipeline may still download datasets and auxiliary models. Register
and apply the appropriate `model`, `api`, or `vllm` marker to new integration tests.
A marker alone does not prevent import-time work: keep integration files outside
`test/unit`.

The separate **Integration tests** workflow runs model and API jobs on pushes to
`main`, weekly, and via **Run workflow**. Use its suite selector to run either job
individually. The `vllm` selection runs only on manual dispatch and requires a
self-hosted Linux x64 runner labeled `gpu`, with a compatible NVIDIA driver. It
is not scheduled or run on pull requests. The larger benchmark and vision suites
remain local opt-in tests as described in `test/local/README.md`.
