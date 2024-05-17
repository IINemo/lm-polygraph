# Description: IsotonicPCC normalizer for UE values

import pickle

import numpy as np
from cir_model import CenteredIsotonicRegression

from lm_polygraph.normalizers.base import BaseUENormalizer

class IsotonicPCCNormalizer(BaseUENormalizer):
    def __init__(self):
        self.scaler = None

    def fit(self, gen_metrics: np.ndarray, ues: np.ndarray) -> None:
        """Fits centered isotonic regression to the gen_metrics and ues data."""
        self.scaler = CenteredIsotonicRegression(out_of_bounds='clip',
                                         increasing=False,
                                         y_min=0, y_max=1)
        self.scaler.fit(ues, gen_metrics)


    def transform(self, ues: np.ndarray) -> np.ndarray:
        """Transforms the ues data using the fitted CenteredIsotonicRegression."""
        return self.scaler.transform(ues[:, None]).squeeze()

    def dumps(self) -> str:
        """Dumps the CenteredIsotonicRegression object to a string."""
        return pickle.dumps(self.scaler)

    @staticmethod
    def loads(scaler):
        """Loads the CenteredIsotonicRegression object from a string."""
        normalizer = IsotonicPCCNormalizer()
        normalizer.scaler = pickle.loads(scaler)
        return normalizer
