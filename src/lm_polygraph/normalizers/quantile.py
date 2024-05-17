# Description: Quantile normalizer for UE values

import pickle

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from lm_polygraph.normalizers.base import BaseUENormalizer


class QuantileNormalizer(BaseUENormalizer):
    def __init__(self):
        self.scaler = None

    def fit(self, ues: np.ndarray) -> None:
        """Fits QuantileTransformer to the gen_metrics and ues data."""
        self.scaler = QuantileTransformer(output_distribution="uniform")
        conf = -ues
        self.scaler.fit(conf[:, None])

    def transform(self, ues: np.ndarray) -> np.ndarray:
        """Transforms the ues data using the fitted QuantileTransformer."""
        conf = -ues
        return self.scaler.transform(conf[:, None]).squeeze()

    def dumps(self) -> str:
        """Dumps the QuantileNormalizer object to a string."""
        return pickle.dumps(self.scaler)

    @staticmethod
    def loads(scaler):
        """Loads the QuantileNormalizer object from a string."""
        normalizer = QuantileNormalizer()
        normalizer.scaler = pickle.loads(scaler)
        return normalizer
