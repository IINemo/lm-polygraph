import os
import pandas as pd
import numpy as np
import logging
import requests
import io

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List, Union, Optional
from PIL import Image

log = logging.getLogger("lm_polygraph")


class Dataset:
    """
    Seq2seq or vision-language dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(
        self, x: List[str], y: List[str], batch_size: int, images: Optional[str] = None
    ):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
        """
        self.x = x
        self.y = y
        self.images = images
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str], Optional[List]]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str]]]: iterates over batches in dataset,
                returns list of input texts and list of corresponding output texts.
        """
        for i in range(0, len(self.x), self.batch_size):
            batch_x = self.x[i : i + self.batch_size]
            batch_y = self.y[i : i + self.batch_size]
            batch_images = (
                self.images[i : i + self.batch_size]
                if self.images is not None
                else None
            )
            yield (batch_x, batch_y, batch_images)

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
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        if self.images is not None:
            self.images = [self.images[i] for i in indices]
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
            np.array(self.x),
            np.array(self.y),
            test_size=test_size,
            random_state=seed,
        )
        if self.images is not None:
            images_train, images_test = train_test_split(
                np.array(self.images), test_size=test_size, random_state=seed
            )
        else:
            images_train = images_test = None

        if split == "train":
            self.x, self.y, self.images = (
                X_train.tolist(),
                y_train.tolist(),
                images_train.tolist() if images_train is not None else None,
            )
        else:
            self.x, self.y, self.images = (
                X_test.tolist(),
                y_test.tolist(),
                images_test.tolist() if images_test is not None else None,
            )

        return (
            X_train.tolist(),
            X_test.tolist(),
            y_train.tolist(),
            y_test.tolist(),
        )

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
        **kwargs,
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
    def load_hf_dataset(
        path: Union[str, List[str]],
        split: str,
        **kwargs,
    ):
        load_from_disk = kwargs.pop("load_from_disk", False)
        if load_from_disk:
            dataset_name = path
            dataset = hf_dataset.load_from_disk(path)
        elif isinstance(path, str):
            dataset_name = path
            dataset = load_dataset(path, split=split, **kwargs)
        else:
            dataset_name = path[0]
            dataset = load_dataset(*path, split=split, **kwargs)

        return dataset_name, dataset

    @staticmethod
    def from_datasets(
        dataset_path: Union[str, List[str]],
        x_column: str,
        y_column: str,
        batch_size: int,
        im_column: Optional[str] = None,
        prompt: str = "",
        description: str = "",
        mmlu_max_subject_size: int = 100,
        n_shot: int = 0,
        few_shot_split: str = "train",
        few_shot_prompt: Optional[str] = None,
        instruct: bool = False,
        split: str = "test",
        size: int = None,
        **kwargs,
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
        dataset_name, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)

        if size is not None and size < len(dataset):
            dataset = dataset.select(range(size))

        if "allenai/c4" in dataset_name.lower():
            x, y = [], []
            for inst in dataset:
                if len(inst[x_column]) <= 1024:
                    x.append(inst[x_column])
                    y.append(inst[y_column])
        else:
            x = dataset[x_column]
            if y_column is not None:
                y = dataset[y_column]
            else:
                y = ["" for _ in range(len(x))]

        images = dataset[im_column] if im_column else None

        return Dataset(x, y, batch_size, images=images)

    @staticmethod
    def load(path_or_path_and_files: Union[str, List[str]], *args, **kwargs):
        """
        Creates the dataset from either local .csv path (if such exists) or Huggingface datasets.
        See `from_csv` and `from_datasets` static functions for the description of *args and **kwargs arguments.

        Parameters:
            path_or_path_and_files (str or List[str]): local path to .csv table or HF path to dataset.
        """
        if isinstance(path_or_path_and_files, str) and os.path.isfile(
            path_or_path_and_files
        ):
            return Dataset.from_csv(path_or_path_and_files, *args, **kwargs)
        return Dataset.from_datasets(path_or_path_and_files, *args, **kwargs)

    @staticmethod
    def get_images(images: List[Union[Image.Image, str, bytes]]):
        imgs: List[Image.Image] = []
        for image_input in images:
            try:
                if isinstance(image_input, Image.Image):
                    imgs.append(image_input.convert("RGB"))
                elif isinstance(image_input, str) and image_input.startswith("http"):
                    response = requests.get(image_input, stream=True, timeout=10)
                    response.raise_for_status()
                    imgs.append(Image.open(io.BytesIO(response.content)).convert("RGB"))
                elif isinstance(image_input, str):
                    imgs.append(Image.open(image_input).convert("RGB"))
                elif isinstance(image_input, (bytes, bytearray)):
                    imgs.append(Image.open(io.BytesIO(image_input)).convert("RGB"))
                else:
                    log.warning(f"Unsupported image input format: {type(image_input)}")
            except Exception as e:
                log.warning(f"Failed to load image '{image_input}': {e}")
        return imgs
