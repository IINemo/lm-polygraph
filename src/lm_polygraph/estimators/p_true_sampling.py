import numpy as np

from typing import Dict

from .estimator import Estimator


class PTrueSampling(Estimator):
    def __init__(self):
        super().__init__(['p_true_sampling'], 'sequence')

    def __str__(self):
        return 'PTrueSampling'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        pue = stats['p_true_sampling']
        return -np.array(pue)
