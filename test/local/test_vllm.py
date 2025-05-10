import subprocess
import pathlib


# ================= TEST HELPERS ==================


def exec_bash(s):
    return subprocess.run(s, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


def run_config_with_overrides(config_name, **overrides):
    command = f"HYDRA_CONFIG={pwd()}/../configs/{config_name}.yaml polygraph_eval"
    for key, value in overrides.items():
        command += f" {key}='{value}'"
    print(command)
    exec_result = exec_bash(command)

    assert exec_result.returncode == 0, f"running {command} failed!"

    return exec_result


# ================= TEST CASES ==================


def test_vllm():
    exec_result = run_config_with_overrides("test_polygraph_eval_vllm")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"
