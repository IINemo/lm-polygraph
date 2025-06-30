import os
import logging

from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.stat_calculators.train_mtopdiv import (
    TrainMTopDivCalculator,
)


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
        few_shot_split=args.few_shot_split,
        few_shot_prompt=None,
        instruct=None,
        split=args.train_split,
        size=args.size,
        load_from_disk=args.load_from_disk,
        trust_remote_code=False,
    )

    if args.subsample_train_dataset != -1:
        train_dataset.subsample(
            args.subsample_train_dataset,
            seed=(
                int(list(args.seed)[0]) if not isinstance(args.seed, int) else args.seed
            ),
        )

    log.info("Done loading train data.")
    return train_dataset


def load_stat_calculator(config, builder):
    priority = config.heads_extraction_priority
    cache_path = os.path.join(config.cache_path, config.model_heads_cache)
    max_heads = config.max_heads
    n_jobs = config.n_jobs
    load_train_dataset_fn = lambda: load_dataset(config)

    return TrainMTopDivCalculator(
        priority,
        load_train_dataset_fn,
        cache_path,
        max_heads,
        n_jobs,
    )
