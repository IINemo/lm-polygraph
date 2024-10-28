import subprocess
import pathlib
import os
import torch

from lm_polygraph.utils.manager import UEManager


# ================= TEST HELPERS ==================


def run_eval(dataset):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    command = f"HYDRA_CONFIG={pwd()}/../../examples/configs/polygraph_eval_{dataset}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                subsample_train_dataset=2 \
                subsample_background_train_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={device} \
                save_path={pwd()}"

    return subprocess.run(command, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


def check_result(exec_result):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    assert len(man.estimations[("sequence", "MaximumSequenceProbability")]) == 2

    os.remove(f"{pwd()}/ue_manager_seed1")


# ================= TEST CASES ==================


def test_coqa():
    exec_result = run_eval("coqa")
    check_result(exec_result)


def test_triviaqa():
    exec_result = run_eval("triviaqa")
    check_result(exec_result)


def test_mmlu():
    exec_result = run_eval("mmlu")
    check_result(exec_result)


def test_gsm8k():
    exec_result = run_eval("gsm8k")
    check_result(exec_result)


def test_wmt14_fren():
    exec_result = run_eval("wmt14_fren")
    check_result(exec_result)


def test_wmt19_deen():
    exec_result = run_eval("wmt19_deen")
    check_result(exec_result)


def test_xsum():
    exec_result = run_eval("xsum")
    check_result(exec_result)
