import os
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict

from typing import Iterable, Tuple, List

class Dataset:
    def __init__(self, x: List[str], y: List[str], batch_size: int):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str]]]:
        for i in range(0, len(self.x), self.batch_size):
            yield self.x[i:i + self.batch_size], self.y[i:i + self.batch_size]

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size
    
    def select(self, indices: List[int]):
        self.x = np.array(self.x)[indices].tolist()
        self.y = np.array(self.y)[indices].tolist()
        return self
    
    def train_test_split(self, test_size: int, seed: int, split: str = "train"):
        X_train, X_test, y_train, y_test = train_test_split(np.array(self.x), np.array(self.y), test_size=test_size, random_state=seed)
            
        if split == "train":
            self.x = X_train.tolist()
            self.y = y_train.tolist()
        else:
            self.x = X_test.tolist()
            self.y = y_test.tolist()
            
        return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()
    
    def subsample(self, size: int, seed: int):
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(csv_path: str, x_column: str, y_column: str, batch_size: int, **kwargs):
        csv = pd.read_csv(csv_path)
        x = csv[x_column].tolist()
        y = csv[y_column].tolist()
        return Dataset(x, y, batch_size)

    @staticmethod
    def from_datasets(csv_path: str, x_column: str, y_column: str, batch_size: int, prompt: str):
        dataset: DatasetDict = load_dataset(csv_path)['test']
        x = dataset[x_column].tolist()
        x = [prompt.strip() + " " + text for text in x]
        y = dataset[y_column].tolist()
        return Dataset(x, y, batch_size)

    @staticmethod
    def load(csv_path, *args, **kwargs):
        if os.path.exists(csv_path):
            return Dataset.from_csv(csv_path, *args, **kwargs)
        return Dataset.from_datasets(csv_path, *args, **kwargs)