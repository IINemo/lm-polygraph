import os
import subprocess


def exec_bash(s):
    return subprocess.run(s, shell=True)


def test_just_works_hydra():
    command = (
        "HYDRA_CONFIG=../test/configs/test_polygraph_eval.yaml scripts/polygraph_eval"
    )
    exec_result = exec_bash(command)
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"
    os.remove("ue_manager_seed1")
