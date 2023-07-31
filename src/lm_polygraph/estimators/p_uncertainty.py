import numpy as np

from typing import Dict

from .estimator import Estimator


class PUncertainty(Estimator):
    def __init__(self):
        super().__init__(['p_uncertainty'], 'sequence')

    def __str__(self):
        return 'PUncertainty'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        pue = stats['p_uncertainty']
        return -np.array(pue)
