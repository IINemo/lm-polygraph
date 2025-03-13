import subprocess
import pathlib
import os
import torch
import json
import pytest
import diskcache as dc

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)


# ================= TEST HELPERS ==================


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture(scope="module")
def reference():
    with open(f"{pwd()}/fixtures/input_output_fixtures.json") as f:
        return json.load(f)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


def check_result(dataset, exec_result, reference, method=None):
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"

    man = UEManager.load(
        f"{pwd()}/ue_manager_seed1",
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(None),
        available_stat_calculators=register_default_stat_calculators(
            model_type="Whitebox"
        ),
    )

    if method is None:
        assert len(man.estimations[("sequence", "MaximumSequenceProbability")]) == 2

    key = dataset
    if method:
        key += f"_{method}"

    assert man.stats["input_texts"][0] == reference[key + "_input"]
    assert man.stats["target_texts"][0] == reference[key + "_output"]

    os.remove(f"{pwd()}/ue_manager_seed1")


# ================= TEST CASES ==================


def run_eval(dataset):
    command = f"HYDRA_CONFIG={pwd()}/../../examples/configs/polygraph_eval_{dataset}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={get_device()} \
                save_path={pwd()} \
                stat_calculators.1.cfg.size=10 \
                stat_calculators.1.cfg.bg_size=20"

    return subprocess.run(command, shell=True)


def test_coqa(reference):
    exec_result = run_eval("coqa")
    check_result("coqa", exec_result, reference)


def test_triviaqa(reference):
    exec_result = run_eval("triviaqa")
    check_result("triviaqa", exec_result, reference)


def test_mmlu(reference):
    exec_result = run_eval("mmlu")
    check_result("mmlu", exec_result, reference)


def test_gsm8k(reference):
    exec_result = run_eval("gsm8k")
    check_result("gsm8k", exec_result, reference)


def test_wmt14_fren(reference):
    exec_result = run_eval("wmt14_fren")
    check_result("wmt14_fren", exec_result, reference)


def test_wmt19_deen(reference):
    exec_result = run_eval("wmt19_deen")
    check_result("wmt19_deen", exec_result, reference)


def test_xsum(reference):
    exec_result = run_eval("xsum")
    check_result("xsum", exec_result, reference)


# ================= INSTRUCT TEST CASES ==================


def run_instruct_eval(dataset, method):
    command = f"HYDRA_CONFIG={pwd()}/../../examples/configs/instruct/polygraph_eval_{dataset}_{method}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model=stablelm-1.6b-chat \
                model.load_model_args.device_map={get_device()} \
                save_path={pwd()}"

    return subprocess.run(command, shell=True)


METHODS = [
    "ling_1s",
    "verb_1s_top1",
    "verb_1s_topk",
    "verb_2s_top1",
    "verb_2s_topk",
    "verb_2s_cot",
    "empirical_baselines",
]


def test_coqa_instruct(reference):
    for method in METHODS:
        exec_result = run_instruct_eval("coqa", method)
        check_result("coqa", exec_result, reference, method)


def test_triviaqa_instruct(reference):
    for method in METHODS:
        exec_result = run_instruct_eval("triviaqa", method)
        check_result("triviaqa", exec_result, reference, method)


def test_mmlu_instruct(reference):
    for method in METHODS:
        exec_result = run_instruct_eval("mmlu", method)
        check_result("mmlu", exec_result, reference, method)


# ================= CLAIM-LEVEL ==================


def run_claim_eval(dataset):
    fixed_cache = dc.Cache(f"{pwd()}/fixtures/openai_chat_cache.diskcache")
    with dc.Cache(
        os.path.expanduser("~") + "/.cache/openai_chat_cache.diskcache"
    ) as cache:
        for k in fixed_cache:
            cache[k] = fixed_cache[k]

    command = f"HYDRA_CONFIG={pwd()}/../../examples/configs/polygraph_eval_{dataset}.yaml \
                polygraph_eval \
                subsample_eval_dataset=2 \
                model.path=bigscience/bloomz-560m \
                model.load_model_args.device_map={get_device()} \
                save_path={pwd()}"

    return subprocess.run(command, shell=True)


def check_claim_level_result(dataset, reference):
    man = UEManager.load(
        f"{pwd()}/ue_manager_seed1",
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(None),
        available_stat_calculators=register_default_stat_calculators(
            model_type="Whitebox"
        ),
    )

    assert man.stats["input_texts"][0] == reference[dataset + "_input"]
    assert man.stats["target_texts"][0] == reference[dataset + "_output"]

    os.remove(f"{pwd()}/ue_manager_seed1")


def test_person_bio(reference):
    base_dataset_name = "person_bio"
    langs = ["en_mistral", "zh"]

    for lang in langs:
        dataset = f"{base_dataset_name}_{lang}"
        run_claim_eval(dataset)
        check_claim_level_result(dataset, reference)
