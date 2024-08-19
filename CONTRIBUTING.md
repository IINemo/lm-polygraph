# Creating a Pull Request

1. Fork the repository
2. Clone the repository to your local machine
3. Create a new branch for your changes
4. Make your changes
5. **Make sure to update documentation if needed**
6. Use flake8 and black to conform to the code style. Github actions will check this automatically using the following command:
    ```bash
    flake8 --extend-ignore E501,F405,F403,E203 --per-file-ignores __init__.py:F401 .
    ```
    Install flake8-black beforehand to check for black formatting as well.
7. Run tests with `pytest` and make sure they pass
8. Commit and push your changes
9. Create a pull request to the main branch of the original repository
