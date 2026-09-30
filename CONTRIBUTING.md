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

## Deterministic numerical tests

Run the estimator and high-level generation tests without model downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest test/test_estimator_numerics.py test/test_high_level_api.py
```

These tests use hand-calculated statistics and a tiny local model with uniform
token probabilities. When adding estimator tests, assert expected scores and
output shapes, and include empty or EOS-only generations where applicable.
The existing `test/test_estimators.py` and benchmark tests additionally exercise
pretrained models and may require downloads.
