import subprocess
import pathlib
import os
import torch
import sys

from lm_polygraph.utils.manager import UEManager


# ================= TEST HELPERS ==================
def load_input_texts(dataset):
    with open(f"{pwd()}/{dataset}.txt", "r") as f:
        return f.read()


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


def check_result(dataset, exec_result):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    assert len(man.estimations[("sequence", "MaximumSequenceProbability")]) == 2
    try:
        assert man.stats["input_texts"][0] == load_input_texts(dataset)
    except:
        sys.stdout.flush()
        breakpoint()

    os.remove(f"{pwd()}/ue_manager_seed1")


# ================= TEST CASES ==================


#def test_coqa():
#    exec_result = run_eval("coqa")
#    check_result('coqa', exec_result)


#def test_triviaqa():
#    exec_result = run_eval("triviaqa")
#    check_result('triviaqa', exec_result)


#def test_mmlu():
#    exec_result = run_eval("mmlu")
#    check_result('mmlu', exec_result)


#def test_gsm8k():
#    exec_result = run_eval("gsm8k")
#    check_result('gsm8k', exec_result)


#def test_wmt14_fren():
#    exec_result = run_eval("wmt14_fren")
#    check_result('wmt14_fren', exec_result)
#
#
#def test_wmt19_deen():
#    exec_result = run_eval("wmt19_deen")
#    check_result('wmt19_deen', exec_result)


def test_xsum():
    exec_result = run_eval("xsum")
    check_result('xsum', exec_result)
