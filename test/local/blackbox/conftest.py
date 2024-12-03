import pytest
from pathlib import Path
import yaml
import shutil
import os
from .mock_openai import MockOpenAIServer


def setup_openai_config(base_url: str):
    """Setup OpenAI configuration using both config file and environment variables"""
    # Save original environment variables
    original_env = {
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "OPENAI_API_TYPE": os.environ.get("OPENAI_API_TYPE"),
    }

    # Set environment variables
    os.environ["OPENAI_BASE_URL"] = f"{base_url}/v1"
    os.environ["OPENAI_API_KEY"] = "not-needed"
    os.environ["OPENAI_API_TYPE"] = "open_ai"

    # Setup config file
    config_dir = Path.home() / ".openai"
    config_dir.mkdir(exist_ok=True)

    config = {"base_url": f"{base_url}/v1", "api_key": "not-needed"}

    config_path = config_dir / "config.yaml"

    # Backup existing config if it exists
    if config_path.exists():
        backup_path = config_path.with_suffix(".yaml.bak")
        shutil.copy2(config_path, backup_path)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return original_env


def cleanup_openai_config(original_env: dict):
    """Cleanup OpenAI configuration and restore original environment"""
    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)  # Remove if it was not originally set
        else:
            os.environ[key] = value  # Restore original value

    # Cleanup config file
    config_dir = Path.home() / ".openai"
    config_path = config_dir / "config.yaml"
    backup_path = config_path.with_suffix(".yaml.bak")

    # Remove test config
    if config_path.exists():
        os.remove(config_path)

    # Restore backup if it exists
    if backup_path.exists():
        shutil.move(backup_path, config_path)
    # If no backup exists and we removed the only config, remove empty directory
    elif config_dir.exists() and not any(config_dir.iterdir()):
        config_dir.rmdir()


@pytest.fixture(scope="session", autouse=True)
def mock_openai_server():
    """Session-wide fixture to setup mock OpenAI server"""
    server = MockOpenAIServer(port=8080, max_retries=10)
    server.configure(fixed_response="This is a test response from mock OpenAI server")
    original_env = None

    try:
        server.start()
        original_env = setup_openai_config(server.base_url)

        yield server

    finally:
        server.stop()
        if original_env is not None:
            cleanup_openai_config(original_env)


def pytest_addoption(parser):
    """Add custom pytest options"""
    parser.addoption(
        "--run-real-openai",
        action="store_true",
        default=False,
        help="run tests against real OpenAI API instead of mock",
    )
    parser.addoption(
        "--mock-response",
        action="store",
        default="This is a test response from mock OpenAI server",
        help="specify custom response for mock server",
    )
    parser.addoption(
        "--mock-port",
        type=int,
        default=8080,
        help="specify starting port for mock server",
    )


def pytest_collection_modifyitems(config, items):
    """Skip only when using real OpenAI"""
    run_real = config.getoption("--run-real-openai")
    if run_real:  # Only skip when running real OpenAI
        skip_mock = pytest.mark.skip(reason="using real OpenAI API")
        for item in items:
            if "test_chat_completion" in item.name:
                item.add_marker(skip_mock)


@pytest.fixture
def mock_response(request, mock_openai_server):
    """Fixture to configure mock server response"""
    response = request.config.getoption("--mock-response")
    mock_openai_server.configure(fixed_response=response)
    return response
