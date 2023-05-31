import pandas as pd

from typing import Iterable, Tuple, List


class Dataset:
    def __init__(self, csv_path: str, x_column: str, y_column: str, batch_size: int):
        csv = pd.read_csv(csv_path)
        self.x = csv[x_column].tolist()
        self.y = csv[y_column].tolist()
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str]]]:
        for i in range(0, len(self.x), self.batch_size):
            yield self.x[i:i + self.batch_size], self.y[i:i + self.batch_size]

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size
