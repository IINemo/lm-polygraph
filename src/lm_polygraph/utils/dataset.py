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
        if 'trivia_qa' in csv_path.lower():
            dataset = load_dataset(csv_path, 'rc.nocontext')['test']
        elif 'babi_qa' in csv_path.lower():
            dataset = load_dataset(csv_path, 'en-10k-qa1')['test']
        else:
            dataset = load_dataset(csv_path)['test']
        # In this case this is a NMT dataset
        if 'translation' in dataset.column_names:
            x, y = [], []
            for inst in dataset['translation']:
                x.append(inst[x_column])
                y.append(inst[y_column])
        # For COQA dataset
        elif 'coqa' in csv_path.lower():
            x, y = [], []
            for inst in dataset:
                for question, answer in zip(inst['questions'], inst['answers']['input_text']):
                    x.append(f'Story:\n{inst["story"]}\n\nQuestion:\n{question}\n\nAnswer:\n')
                    y.append(answer)
        # For Babi_QA dataset
        elif 'babi_qa' in csv_path.lower():
            x, y = [], []
            prompt = 'Answer the question given a context. Output only the full name of the location.\n\nExample:\n\nContext:\nMary moved to the bathroom. John went to the hallway. Daniel went back to the hallway. Sandra moved to the garden. John moved to the office. Sandra journeyed to the bathroom. Mary moved to the hallway. Daniel travelled to the office. John went back to the garden. John moved to the bedroom.\nQuestion:\nWhere is Sandra?\nAnswer:\nbathroom\n\n'
            for inst in dataset:
                inst = inst['story']
                context = ''
                for text, answer in zip(inst['text'], inst['answer']):
                    if answer == '':
                        context += text + ' '
                    else:
                        input = prompt + f'Context:\n{context.strip()}\n\nQuestion:\n{text}'
                        x.append(input)
                        y.append(answer)
        # Otherwise, it is a standard one (e.g. summarization)
        else:
            x = [prompt.strip() + " " + text for text in dataset[x_column]]
            y = dataset[y_column]
        return Dataset(x, y, batch_size)

    @staticmethod
    def load(csv_path, *args, **kwargs):
        if os.path.exists(csv_path):
            return Dataset.from_csv(csv_path, *args, **kwargs)
        return Dataset.from_datasets(csv_path, *args, **kwargs)