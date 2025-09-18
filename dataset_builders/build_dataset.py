import datasets

from builders.base import CONFIG as base_config
from builders.babi_qa import CONFIG as babi_qa_config
from builders.coqa import CONFIG as coqa_config
from builders.mmlu import CONFIG as mmlu_config
from builders.person import CONFIG as person_config
from builders.trivia_qa import CONFIG as trivia_qa_config
from builders.wiki import CONFIG as wiki_config
from builders.wmt import CONFIG as wmt_config
from builders.truthfulqa import CONFIG as truthfulqa_config
from builders.samsum import CONFIG as samsum_config
from builders.xsum import CONFIG as xsum_config
from builders.gsm8k import CONFIG as gsm8k_config

DATASET_CONFIG = (
    base_config
    | babi_qa_config
    | coqa_config
    | mmlu_config
    | person_config
    | trivia_qa_config
    | wiki_config
    | wmt_config
    | truthfulqa_config
    | samsum_config
    | xsum_config
    | gsm8k_config
)


def build_dataset(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    if isinstance(config["name"], list):
        dataset = datasets.load_dataset(
            *config["name"], trust_remote_code=True, num_proc=4
        )
    else:
        dataset = datasets.load_dataset(
            config["name"], trust_remote_code=True, num_proc=4
        )

    def prepare_dataset(split):
        x, y = config["prepare_func"](dataset=dataset[config[f"{split}_split"]])
        result_dataset = datasets.Dataset.from_dict({"input": x, "output": y})
        return result_dataset

    result = {}
    if "train_split" in config:
        result["train"] = prepare_dataset("train")
    if "test_split" in config:
        result["test"] = prepare_dataset("test")
    return datasets.DatasetDict(result)
