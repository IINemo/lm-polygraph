import numpy as np

from typing import Dict

from .estimator import Estimator


class Ling1S(Estimator):
    def __init__(self):
        super().__init__([f"ling_1s_response"], "sequence")

    def __str__(self):
        return f"Ling1S"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # parse the text
        return np.array([])
