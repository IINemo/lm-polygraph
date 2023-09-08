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
    def from_datasets(
        csv_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        split: str = "test",
        size: int = None,
        **kwargs
    ):
        dataset = load_dataset(csv_path, split=split, **kwargs)
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
                    #f"Translate from {source_lang} into {target_lang}:\n{inst[x_column]}\nTranslation:\n"
                y.append(inst[y_column])
        else:
            if len(prompt):
                x = [prompt.format(text=text)
                for text in dataset[x_column]]
            else:
                x = dataset[x_column]
            y = dataset[y_column]
        
        if len(prompt):
            x = [prompt.format(text=text)
                for text in dataset[x_column]]
        else:
            x = dataset[x_column]
        y = dataset[y_column]
        
        # if "coqa" in csv_path and split == "test":
        #     split = "validation"
        # if "trivia_qa" in csv_path.lower():
        #     dataset = load_dataset(csv_path, "rc.nocontext", split=split, **kwargs)
        # elif "babi_qa" in csv_path.lower():
        #     dataset = load_dataset(csv_path, "en-10k-qa1", split=split, **kwargs)
        # elif "wmt" in csv_path.lower():
        #     dataset_subset = "de-en" if "de" in [x_column, y_column] else "fr-en"
        #     dataset = load_dataset(csv_path, dataset_subset, split=split, **kwargs)
        # else:
        #     dataset = load_dataset(csv_path, split=split, **kwargs)
        # if size is not None and size < len(dataset):
        #     dataset = dataset.select(range(size))
        # # In this case this is a NMT dataset
        # if "translation" in dataset.column_names:
        #     x, y = [], []
        #     source_lang = (
        #         "German"
        #         if x_column == "de"
        #         else "French"
        #         if x_column == "fr"
        #         else "English"
        #     )
        #     target_lang = (
        #         "German"
        #         if y_column == "de"
        #         else "French"
        #         if y_column == "fr"
        #         else "English"
        #     )
        #     for inst in dataset["translation"]:
        #         x.append(
        #             prompt.format(source_lang=source_lang, target_lang=target_lang, text=inst[x_column])
        #         )
        #             #f"Translate from {source_lang} into {target_lang}:\n{inst[x_column]}\nTranslation:\n"
        #         y.append(inst[y_column])
        #     max_new_tokens = None
        # # For COQA dataset
        # elif "coqa" in csv_path.lower():
        #     x, y = [], []
        #     for inst in dataset:
        #         for question, answer in zip(
        #             inst["questions"], inst["answers"]["input_text"]
        #         ):
        #             x.append(
        #                 f'Answer the question given a story.\nStory:\n{inst["story"]}\n\nQuestion:\n{question}\n\nAnswer:\n'
        #             )
        #             y.append(answer)
        #     max_new_tokens = 14
        # # For Babi_QA dataset
        # elif "babi_qa" in csv_path.lower():
        #     x, y = [], []
        #     prompt = "Answer the question given a context. You must only output the full name of the location the same way it is mentioned in the text. Don't output articles or any additional information.\n\nExample:\n\nContext:\nMary moved to the bathroom. John went to the hallway. Daniel went back to the hallway. Sandra moved to the garden. John moved to the office. Sandra journeyed to the bathroom. Mary moved to the hallway. Daniel travelled to the office. John went back to the garden. John moved to the bedroom.\nQuestion:\nWhere is Sandra?\nAnswer:\nbathroom\n\n"
        #     for inst in dataset:
        #         inst = inst["story"]
        #         context = ""
        #         for text, answer in zip(inst["text"], inst["answer"]):
        #             if answer == "":
        #                 context += text + " "
        #             else:
        #                 input = (
        #                     prompt + f"Context:\n{context.strip()}\n\nQuestion:\n{text}\nAnswer:\n"
        #                 )
        #                 x.append(input)
        #                 y.append(answer)
        #     max_new_tokens = 3
        # # Otherwise, it is a standard one (e.g. summarization)
        # else:
        #     if csv_path == "xsum":
        #         x = [
        #             f"Summarize the text in a one-sentence headline.\n\nText:\n{text}\n\nSummary (one sentence):\n"
        #             for text in dataset[x_column]
        #         ]
        #         max_new_tokens = 42
        #     elif csv_path == "aeslc":
        #         x = [
        #             f"Write a headline for the email.\n\nEmail:\n{text}\n\nHeadline:\n"
        #             for text in dataset[x_column]
        #         ]
        #         max_new_tokens = 20
        #     else:
        #         x = [prompt.strip() + " " + text for text in dataset[x_column]]
        #         max_new_tokens = 200
        #     y = dataset[y_column]

        return Dataset(x, y, batch_size)

    @staticmethod
    def load(csv_path, *args, **kwargs):
        if os.path.exists(csv_path):
            return Dataset.from_csv(csv_path, *args, **kwargs)
        return Dataset.from_datasets(csv_path, *args, **kwargs)