import os
import pathlib
import subprocess
import torch
import json

from lm_polygraph.utils.manager import UEManager


def pwd():
    return pathlib.Path(__file__).parent.resolve()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


result = {}

# ============ CONTINUATION ============


def run_eval(dataset):
    command = f"HYDRA_CONFIG={pwd()}/../../../examples/configs/polygraph_eval_{dataset}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={get_device()} \
                save_path={pwd()} \
                use_density_based_ue=false"

    return subprocess.run(command, shell=True)


def print_result(dataset, exec_result):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    result[f"{dataset}_input"] = man.stats["input_texts"][0]
    result[f"{dataset}_output"] = man.stats["target_texts"][0]

    os.remove(f"{pwd()}/ue_manager_seed1")


datasets = [
    "coqa",
    "triviaqa",
    "mmlu",
    "gsm8k",
    "wmt14_fren",
    "wmt19_deen",
    "xsum",
    "aeslc",
    "babiqa",
]


for dataset in datasets:
    exec_result = run_eval(dataset)
    print_result(dataset, exec_result)

# ============ INSTRUCT ============


datasets = ["coqa", "triviaqa", "mmlu"]
methods = [
    "ling_1s",
    "verb_1s_top1",
    "verb_1s_topk",
    "verb_2s_top1",
    "verb_2s_topk",
    "verb_2s_cot",
    "empirical_baselines",
]


def run_instruct_eval(dataset, method):
    command = f"HYDRA_CONFIG={pwd()}/../../../examples/configs/instruct/polygraph_eval_{dataset}_{method}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model=stablelm-1.6b-chat \
                model.load_model_args.device_map={get_device()} \
                save_path={pwd()} \
                use_density_based_ue=false \
                use_seq_ue=false"

    return subprocess.run(command, shell=True)


def print_instruct_result(dataset, exec_result, method):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    result[f"{dataset}_{method}_input"] = man.stats["input_texts"][0]
    result[f"{dataset}_{method}_output"] = man.stats["target_texts"][0]

    os.remove(f"{pwd()}/ue_manager_seed1")


for dataset in datasets:
    for method in methods:
        exec_result = run_instruct_eval(dataset, method)
        print_instruct_result(dataset, exec_result, method)


# ============ CLAIM ============


datasets = [
    "person_bio_en",
    "person_bio_ru",
    "person_bio_ar",
    "person_bio_zh",
    "wiki_bio",
]


def run_claim_eval(dataset):
    command = f"HYDRA_CONFIG={pwd()}/../../../examples/configs/polygraph_eval_{dataset}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={get_device()} \
                save_path={pwd()}"

    return subprocess.run(command, shell=True)


def print_claim_result(dataset, exec_result):
    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    result[f"{dataset}_input"] = man.stats["input_texts"][0]
    result[f"{dataset}_output"] = man.stats["target_texts"][0]

    os.remove(f"{pwd()}/ue_manager_seed1")


for dataset in datasets:
    exec_result = run_claim_eval(dataset)
    print_claim_result(dataset, exec_result)


with open("input_output_fixtures.json", "w") as f:
    json.dump(result, f)
