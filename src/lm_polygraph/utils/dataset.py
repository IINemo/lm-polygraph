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
            if size < 1:
                size = int(size * len(self.x))
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(csv_path: str, x_column: str, y_column: str, batch_size: int, **kwargs):
        csv = pd.read_csv(csv_path)
        x = csv[x_column].tolist()
        y = csv[y_column].tolist()
        return Dataset(x, y, batch_size)

    @staticmethod
    def from_datasets(
        dataset_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        split: str = "test",
        size: int = None,
        **kwargs
    ):
        dataset = load_dataset(dataset_path, split=split, **kwargs)
        
        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))
            
        if "translation" in dataset.column_names:
            x, y = [], []
            source_lang = (
                "German"
                if x_column == "de"
                else "French"
                if x_column == "fr"
                else "English"
            )
            target_lang = (
                "German"
                if y_column == "de"
                else "French"
                if y_column == "fr"
                else "English"
            )
            for inst in dataset["translation"]:
                x.append(
                    prompt.format(source_lang=source_lang, target_lang=target_lang, text=inst[x_column])
                )
                y.append(inst[y_column])
        elif ("coqa" in dataset_path.lower()) and len(prompt):
            x, y = [], []
            for inst in dataset:
                for question, answer in zip(
                    inst[x_column], inst[y_column]["input_text"]
                ):
                    x.append(
                        prompt.format(story=inst["story"], question=question)
                    )
                    y.append(answer)
        elif ("babi_qa" in dataset_path.lower()) and len(prompt):
            x, y = [], []
            for inst in dataset:
                inst = inst["story"]
                context = ""
                for question, answer in zip(inst[x_column], inst[y_column]):
                    if answer == "":
                        context += text + " "
                    else:
                        x.append(prompt.format(context=context.strip(), question=question))
                        y.append(answer)
        elif len(prompt):
            x = [prompt.format(text=text) for text in dataset[x_column]]
            y = dataset[y_column]
        else:
            x = dataset[x_column]
            y = dataset[y_column]
        
        return Dataset(x, y, batch_size)

    @staticmethod
    def load(csv_path, *args, **kwargs):
        if os.path.exists(csv_path):
            return Dataset.from_csv(csv_path, *args, **kwargs)
        return Dataset.from_datasets(csv_path, *args, **kwargs)