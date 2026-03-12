import numpy as np

from typing import Dict

from .estimator import Estimator


class RandomBaseline(Estimator):
    """
    Sequence-level random baseline that returns scores sampled from Uniform(0, 1).
    """

    def __init__(self):
        super().__init__([], "sequence")

    def __str__(self):
        return "RandomBaseline"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_size = len(stats["input_texts"])
        return np.random.uniform(low=0.0, high=1.0, size=batch_size)
