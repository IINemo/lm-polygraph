import pytest
import subprocess
import pathlib
import time
from openai import OpenAI
from lm_polygraph.utils.manager import UEManager


def exec_bash(s):
    return subprocess.run(s, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.parent.parent.resolve()


def load_test_manager():
    return UEManager.load(f"{pwd()}/../workdir/output/test/ue_manager_seed1")


def run_config_with_overrides(config_name, **overrides):
    command = f"HYDRA_CONFIG={pwd()}/configs/{config_name}.yaml polygraph_eval"
    for key, value in overrides.items():
        command += f" {key}='{value}'"
    print(command)
    exec_result = exec_bash(command)

    assert exec_result.returncode == 0, f"running {command} failed!"

    return exec_result


def test_chat_completion(mock_openai_server):
    """Test chat completion with mock server"""
    client = OpenAI()
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                temperature=0.7,
                max_tokens=50,
            )

            content = response.choices[0].message.content.strip()
            assert (
                content == "This is a test response from mock OpenAI server"
            ), "Unexpected response content"
            break

        except Exception as e:
            if attempt == max_retries - 1:
                pytest.fail(f"All attempts failed: {str(e)}")
            time.sleep(retry_delay)


def test_blackbox_local():
    exec_result = run_config_with_overrides("test_polygraph_eval_blackbox")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"
