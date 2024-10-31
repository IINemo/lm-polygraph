import os
import pathlib
import subprocess
import torch
from lm_polygraph.utils.manager import UEManager

def run_eval(dataset):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    command = f"HYDRA_CONFIG={pwd()}/examples/configs/polygraph_eval_{dataset}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={device} \
                save_path={pwd()} \
                use_density_based_ue=false \
                use_seq_ue=false"

    return subprocess.run(command, shell=True)


def run_verb_eval(dataset, method):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    command = f"HYDRA_CONFIG={pwd()}/examples/configs/instruct/polygraph_eval_{dataset}_{method}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={device} \
                save_path={pwd()} \
                use_density_based_ue=false \
                use_seq_ue=false"
    breakpoint()
    return subprocess.run(command, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


#def print_result(dataset, exec_result):
#    assert (
#        exec_result.returncode == 0
#    ), f"polygraph_eval returned code {exec_result.returncode} != 0"
#
#    man = UEManager.load(f"{pwd()}/ue_manager_seed1")
#
#    os.remove(f"{pwd()}/ue_manager_seed1")

#datasets = ["coqa", "triviaqa", "mmlu", "gsm8k", "wmt14_fren", "wmt19_deen", "xsum"]
#
#for dataset in datasets:
#    exec_result = run_eval(dataset)
#    os.rename(f"{pwd()}/1.txt", f"{pwd()}/{dataset}.txt")

datasets = ["coqa", "triviaqa", "mmlu"]
methods = ["ling_1s",
           "verb_1s_top1",
           "verb_1s_topk",
           "verb_2s_top1",
           "verb_2s_topk",
           "verb_2s_cot",
           "default_instruct"]

def print_verb_result(dataset, exec_result):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(f"{pwd()}/ue_manager_seed1")

    with open(f"{pwd()}/1.txt", "w") as f:
        f.write(man.stats['input_texts'][0])

    os.remove(f"{pwd()}/ue_manager_seed1")

for dataset in datasets:
    for method in methods:
        exec_result = run_verb_eval(dataset, method)
        print_verb_result(dataset, exec_result)
        os.rename(f"{pwd()}/1.txt", f"{pwd()}/{dataset}_{method}.txt")
