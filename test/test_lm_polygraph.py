import os
import subprocess
import pathlib

def exec_bash(s):
    return subprocess.run(s, shell=True)


def test_just_works_hydra():
    pwd = pathlib.Path(__file__).parent.resolve()
    command = (
        f"HYDRA_CONFIG={pwd}/configs/test_polygraph_eval.yaml polygraph_eval"
    )
    exec_result = exec_bash(command)
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"
