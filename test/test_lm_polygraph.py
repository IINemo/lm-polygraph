import subprocess
import pathlib

from lm_polygraph.utils.manager import UEManager

# from lm_polygraph.estimators.ensemble_token_measures import all_token_estimators
# from lm_polygraph.estimators.ensemble_sequence_measures import (
#    all_ep_estimators,
#    all_pe_estimators,
# )


# ================= TEST HELPERS ==================


def exec_bash(s):
    return subprocess.run(s, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


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


# ================= TEST CASES ==================


def test_just_works():
    exec_result = run_config_with_overrides("test_polygraph_eval")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"


def test_all_seq_ue():
    exec_result = run_config_with_overrides("test_polygraph_eval_seq_ue")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"
