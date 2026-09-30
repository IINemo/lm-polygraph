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

## Building the documentation

The user guide lives in `docs/` as reStructuredText; the landing-page quick start
also appears in `README.md`. Keep both examples consistent when changing the API.
Sphinx generates the API reference from `src/lm_polygraph` during the build.

From the repository root, with your virtual environment active:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to preview the result. Review build warnings,
check links to example notebooks, and verify code examples against the current
API. Generated files under `docs/api/` and `docs/_build/` should not be committed.
