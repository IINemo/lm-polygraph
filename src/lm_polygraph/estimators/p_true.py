import numpy as np

from typing import Dict

from .estimator import Estimator


class PTrue(Estimator):
    def __init__(self):
        super().__init__(["p_true"], "sequence")

    def __str__(self):
        return "PTrue"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ptrue = stats["p_true"]
        return -np.array(ptrue)
