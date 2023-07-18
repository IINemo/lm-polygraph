import numpy as np

from typing import Dict

from .estimator import Estimator


def aggregate(ue: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights * ue).sum(-1)

class EPTtu(Estimator):
    def __init__(self):
        super().__init__(['ensemble_token_scores'], 'sequence')

    def __str__(self):
        return 'EPTtu'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        token_level_data = stats['ensemble_token_scores']

        ue = token_level_data['ep_token_level_scores']['total_uncertainty']
        weights = token_level_data['weights']

        ue = aggregate(ue, weights)

        return np.array(ue)
