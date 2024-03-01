import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List


class Dataset:
    """
    Seq2seq dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(self, x: List[str], y: List[str], batch_size: int):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str]]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str]]]: iterates over batches in dataset,
                returns list of input texts and list of corresponding output texts.
        """
        for i in range(0, len(self.x), self.batch_size):
            yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size]

    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to left in the dataset.Must have the same length as input texts.
        """
        self.x = np.array(self.x)[indices].tolist()
        self.y = np.array(self.y)[indices].tolist()
        return self

    def train_test_split(self, test_size: int, seed: int, split: str = "train"):
        """
        Samples dataset into train and test parts.

        Parameters:
            test_size (int): size of test dataset,
            seed (int): seed to perform random splitting with,
            split (str): either 'train' or 'test'. If 'train', lefts only train data in the current dataset object.
                If 'test', left only test data. Default: 'train'.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: train input and target texts list,
                test input and target texts list.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.x), np.array(self.y), test_size=test_size, random_state=seed
        )

        if split == "train":
            self.x = X_train.tolist()
            self.y = y_train.tolist()
        else:
            self.x = X_test.tolist()
            self.y = y_test.tolist()

        return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            if size < 1:
                size = int(size * len(self.x))
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(
        csv_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        **kwargs
    ):
        """
        Creates the dataset from .CSV table.

        Parameters:
            csv_path (str): path to .csv table,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch.
        """
        csv = pd.read_csv(csv_path)
        x = csv[x_column].tolist()
        y = csv[y_column].tolist()

        if len(prompt):
            x = [prompt.format(text=text) for text in x]

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
        """
        Creates the dataset from Huggingface datasets.

        Parameters:
            dataset_path (str): HF path to dataset,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch,
            prompt (str): prompt template to use for input texts (default: ''),
            split (str): dataset split to take data from (default: 'text'),
            size (Optional[int]): size to subsample dataset to. If None, the full dataset split will be taken.
                Default: None.
        """
        load_from_disk = kwargs.pop("load_from_disk", False)
        if load_from_disk:
            dataset_name = dataset_path
            dataset = hf_dataset.load_from_disk(dataset_path)
        elif isinstance(dataset_path, str):
            dataset_name = dataset_path
            dataset = load_dataset(dataset_path, split=split, **kwargs)
        else:
            dataset_name = dataset_path[0]
            dataset = load_dataset(*dataset_path, split=split, **kwargs)

        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))

        if "translation" in dataset.column_names:
            x, y = [], []
            source_lang = (
                "German"
                if x_column == "de"
                else "French" if x_column == "fr" else "English"
            )
            target_lang = (
                "German"
                if y_column == "de"
                else "French" if y_column == "fr" else "English"
            )
            for inst in dataset["translation"]:
                x.append(
                    prompt.format(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        text=inst[x_column],
                    )
                )
                y.append(inst[y_column])
        elif ("coqa" in dataset_name.lower()) and len(prompt):
            x, y = [], []
            for inst in dataset:
                for question, answer in zip(
                    inst[x_column], inst[y_column]["input_text"]
                ):
                    x.append(prompt.format(story=inst["story"], question=question))
                    y.append(answer)
        elif ("babi_qa" in dataset_name.lower()) and len(prompt):
            x, y = [], []
            for inst in dataset:
                inst = inst["story"]
                context = ""
                for text, answer in zip(inst[x_column], inst[y_column]):
                    if answer == "":
                        context += text + " "
                    else:
                        x.append(prompt.format(context=context.strip(), question=text))
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
        """
        Creates the dataset from either local .csv path (if such exists) or Huggingface datasets.
        See `from_csv` and `from_datasets` static functions for the description of *args and **kwargs arguments.

        Parameters:
            csv_path (str): local path to .csv table or HF path to dataset.
        """
        if isinstance(csv_path, str) and os.path.isfile(csv_path):
            return Dataset.from_csv(csv_path, *args, **kwargs)
        return Dataset.from_datasets(csv_path, *args, **kwargs)
