import subprocess
import pathlib
import os
import torch
import sys

from lm_polygraph.utils.manager import UEManager


# ================= TEST HELPERS ==================
def load_input_texts(dataset, method=None):
    filename = f"{dataset}"
    if method:
        filename += f"_{method}"

    with open(f"{pwd()}/{filename}.txt", "r") as f:
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


def run_instruct_eval(dataset, method):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    command = f"HYDRA_CONFIG={pwd()}/../../examples/configs/instruct/polygraph_eval_{dataset}_{method}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                subsample_train_dataset=2 \
                subsample_background_train_dataset=2 \
                model=stablelm-1.6b-chat \
                model.load_model_args.device_map={device} \
                use_density_base_ue=false \
                save_path={pwd()}"

    return subprocess.run(command, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


def check_result(dataset, exec_result, method=None):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    if method is None:
        assert len(man.estimations[("sequence", "MaximumSequenceProbability")]) == 2
    try:
        assert man.stats["input_texts"][0] == load_input_texts(dataset, method)
    except:
        sys.stdout.flush()
        breakpoint()

    os.remove(f"{pwd()}/ue_manager_seed1")


# ================= TEST CASES ==================


def test_coqa():
    exec_result = run_eval("coqa")
    check_result('coqa', exec_result)


def test_triviaqa():
    exec_result = run_eval("triviaqa")
    check_result('triviaqa', exec_result)


def test_mmlu():
    exec_result = run_eval("mmlu")
    check_result('mmlu', exec_result)


def test_gsm8k():
    exec_result = run_eval("gsm8k")
    check_result('gsm8k', exec_result)


def test_wmt14_fren():
    exec_result = run_eval("wmt14_fren")
    check_result('wmt14_fren', exec_result)


def test_wmt19_deen():
    exec_result = run_eval("wmt19_deen")
    check_result('wmt19_deen', exec_result)


def test_xsum():
    exec_result = run_eval("xsum")
    check_result('xsum', exec_result)

# ================= INSTRUCT TEST CASES ==================

METHODS = ["ling_1s", "verb_1s_top1", "verb_1s_topk", "verb_2s_top1", "verb_2s_topk", "verb_2s_cot", "empirical_baselines"]

def test_coqa_instruct():
    for method in METHODS:
        exec_result = run_instruct_eval("coqa", method)
        check_result('coqa', exec_result, method)

def test_triviaqa_instruct():
    for method in METHODS:
        exec_result = run_instruct_eval("triviaqa", method)
        check_result('triviaqa', exec_result, method)

def test_mmlu_instruct():
    for method in METHODS:
        exec_result = run_instruct_eval("mmlu", method)
        check_result('mmlu', exec_result, method)
