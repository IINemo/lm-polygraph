from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.stat_calculators.statistic_extraction import (
    TrainingStatisticExtractionCalculator,
)
import logging

log = logging.getLogger("lm_polygraph")


def load_dataset(args):
    log.info("=" * 100)
    log.info(f"Loading train dataset...")
    if args.train_test_split:
        raise NotImplementedError
    else:
        if (args.train_dataset is not None) and (args.train_dataset != args.dataset):
            dataset_name = args.train_dataset
        else:
            dataset_name = args.dataset

        train_dataset = Dataset.load(
            dataset_name,
            args.text_column,
            getattr(args, "label_column", None),
            batch_size=args.batch_size,
            prompt=getattr(args, "prompt", ""),
            description=getattr(args, "description", ""),
            mmlu_max_subject_size=(
                getattr(args, "mmlu_max_subject_size", 100)
                if "cais/mmlu" in dataset_name
                else 0
            ),
            n_shot=getattr(args, "n_shot", 5),
            few_shot_split=getattr(args, "few_shot_split", "train"),
            few_shot_prompt=getattr(args, "few_shot_prompt", None),
            instruct=getattr(args, "instruct", None),
            split=args.train_split,
            size=10_000,
            load_from_disk=args.load_from_disk,
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )

    background_train_dataset = Dataset.load(
        args.background_train_dataset,
        args.background_train_dataset_text_column,
        args.background_train_dataset_label_column,
        batch_size=args.batch_size,
        data_files=args.background_train_dataset_data_files,
        split="train",
        size=100_000,
        load_from_disk=args.background_load_from_disk,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    if args.subsample_train_dataset != -1:
        train_dataset.subsample(args.subsample_train_dataset, seed=args.seed)
    if args.subsample_background_train_dataset != -1:
        background_train_dataset.subsample(
            args.subsample_background_train_dataset, seed=args.seed
        )

    log.info(f"Done loading train data.")
    return train_dataset, background_train_dataset


def load_stat_calculator(config, builder):
    train_dataset, background_train_dataset = load_dataset(config)
    return TrainingStatisticExtractionCalculator(
        train_dataset, background_train_dataset
    )
