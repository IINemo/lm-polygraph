# Description: MinMax normalizer for UE values

import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from lm_polygraph.normalizers.base import BaseUENormalizer


class MinMaxNormalizer(BaseUENormalizer):
    def __init__(self):
        self.scaler = None

    def fit(self, ues: np.ndarray) -> None:
        """Fits MinMaxScaler to the gen_metrics and ues data."""
        self.scaler = MinMaxScaler(clip=True)
        conf = -ues
        self.scaler.fit(conf[:, None])

    def transform(self, ues: np.ndarray) -> np.ndarray:
        """Transforms the ues data using the fitted MinMaxScaler."""
        conf = -ues
        return self.scaler.transform(conf[:, None]).squeeze()

    def dumps(self) -> str:
        """Dumps the MinMaxScaler object to a string."""
        return pickle.dumps(self.scaler)

    @staticmethod
    def loads(scaler):
        """Loads the MinMaxScaler object from a string."""
        normalizer = MinMaxNormalizer()
        normalizer.scaler = pickle.loads(scaler)
        return normalizer
