import logging
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List, Optional
from lm_polygraph.stat_calculators.train_mtopdiv import (
    TrainMTopDivCalculator,
)

log = logging.getLogger("lm_polygraph")


class Dataset:
    def __init__(self, x: List[str], y: List[str], labels: List[int], batch_size: int):
        self.x = x
        self.y = y
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str], List[int]]]:
        for i in range(0, len(self.x), self.batch_size):
            yield (
                self.x[i : i + self.batch_size],
                self.y[i : i + self.batch_size],
                self.labels[i : i + self.batch_size],
            )

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        return self

    def subsample(self, n_samples: int, seed: Optional[int] = None):
        if n_samples >= len(self.x):
            return self

        indices = list(range(len(self.x)))
        if seed is not None:
            np.random.seed(seed)
        selected_indices = np.random.choice(indices, n_samples, replace=False)
        return self.select(selected_indices.tolist())

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        context_column: str,
        question_column: str,
        prompt_column: str,
        response_column: str,
        label_column: str,
        batch_size: int,
    ):
        def assemble_query(row):
            return row[prompt_column].format(row[context_column], row[question_column])

        csv = pd.read_csv(csv_path)
        x = csv.apply(assemble_query, axis=1).tolist()
        y = csv[response_column].tolist()
        labels = csv[label_column].tolist()
        return cls(x, y, labels, batch_size)


def load_dataset(args):
    log.info("=" * 100)
    log.info("Loading train dataset...")

    train_dataset = Dataset.from_csv(
        csv_path=args.train_dataset,
        context_column=args.context_column,
        question_column=args.question_column,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
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
    train_dataset = load_dataset(config)
    return TrainMTopDivCalculator(train_dataset)
