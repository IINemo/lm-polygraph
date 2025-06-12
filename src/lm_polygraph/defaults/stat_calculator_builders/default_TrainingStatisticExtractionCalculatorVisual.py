from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.stat_calculators.statistic_extraction_visual import (
    TrainingStatisticExtractionCalculatorVisual,
)
import logging

log = logging.getLogger("lm_polygraph")


def load_dataset(args):
    log.info("=" * 100)
    log.info("Loading train dataset...")

    train_dataset = Dataset.load(
        args.dataset,
        args.text_column,
        getattr(args, "label_column", None),
        batch_size=args.batch_size,
        prompt=getattr(args, "prompt", ""),
        description=getattr(args, "description", ""),
        mmlu_max_subject_size=100,
        n_shot=getattr(args, "n_shot", 5),
        im_column=getattr(args, "im_column", None),
        few_shot_split=args.few_shot_split,
        few_shot_prompt=None,
        instruct=None,
        split=args.train_split,
        size=args.size,
        load_from_disk=args.load_from_disk,
        trust_remote_code=False,
    )

    background_train_dataset = Dataset.load(
        args.background_train_dataset,
        args.background_train_dataset_text_column,
        args.background_train_dataset_label_column,
        batch_size=args.batch_size,
        data_files=args.background_train_dataset_data_files,
        im_column=getattr(args, "background_images", None),
        split="train",
        size=args.bg_size,
        load_from_disk=args.background_load_from_disk,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    if args.subsample_train_dataset != -1:
        train_dataset.subsample(
            args.subsample_train_dataset,
            seed=(
                int(list(args.seed)[0]) if not isinstance(args.seed, int) else args.seed
            ),
        )
    if args.subsample_background_train_dataset != -1:
        background_train_dataset.subsample(
            args.subsample_background_train_dataset,
            seed=(
                int(list(args.seed)[0]) if not isinstance(args.seed, int) else args.seed
            ),
        )

    log.info("Done loading train data.")
    return train_dataset, background_train_dataset


def load_stat_calculator(config, builder):
    train_dataset, background_train_dataset = load_dataset(config)
    return TrainingStatisticExtractionCalculatorVisual(
        train_dataset, background_train_dataset
    )
