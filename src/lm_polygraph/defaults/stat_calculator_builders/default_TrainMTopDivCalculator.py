from typing import List
import numpy as np
import pandas as pd

from lm_polygraph.stat_calculators.train_mtopdiv import (
    TrainMTopDivCalculator,
)


class MTopDivHoldoutDataset:
    def __init__(
        self,
        csv_path: str,
        context_column: str,
        question_column: str,
        prompt_column: str,
        response_column: str,
        label_column: str,
        batch_size: int,
        subsample_train_dataset: int,
        seed,
    ):
        self.csv_path = csv_path
        self.context_column = context_column
        self.question_column = question_column
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.subsample_train_dataset = subsample_train_dataset
        self.seed = seed

        self.prompts = []
        self.responses = []
        self.labels = []

    def from_csv(self):
        def assemble_query(row):
            return row[self.prompt_column].format(row[self.context_column], row[self.question_column])

        df = pd.read_csv(self.csv_path)
        self.prompts = df.apply(assemble_query, axis=1).tolist()
        self.responses = df[self.response_column].tolist()
        self.labels = df[self.label_column].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.responses[idx], self.labels[idx]

    def select(self, indices: List[int]):
        self.prompts = [self.prompts[i] for i in indices]
        self.responses = [self.responses[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        return self

    def subsample(self):
        if self.subsample_train_dataset >= len(self):
            return self
        rng = np.random.default_rng(self.seed)
        selected_indices = rng.choice(
            len(self),
            size=self.subsample_train_dataset,
            replace=False,
        ).tolist()
        return self.select(selected_indices)

def load_stat_calculator(config, builder):
    priority = config.heads_extraction_priority
    cache_path = config.cache_path
    max_heads = config.max_heads
    n_jobs = config.n_jobs

    train_dataset = MTopDivHoldoutDataset(
        csv_path=config.train_data_path,
        context_column=config.context_column,
        question_column=config.question_column,
        prompt_column=config.prompt_column,
        response_column=config.response_column,
        label_column=config.label_column,
        batch_size=config.batch_size,
        subsample_train_dataset=config.subsample_train_dataset,
        seed=config.seed,
    )

    return TrainMTopDivCalculator(
        priority,
        train_dataset,
        cache_path,
        max_heads,
        n_jobs,
    )
