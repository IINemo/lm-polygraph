import numpy as np

from typing import Dict

from .estimator import Estimator


class MaxProbabilitySeq(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods'], 'sequence')

    def __str__(self):
        return 'MaxProbabilitySeq'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats['greedy_log_likelihoods']
        return np.array([-np.mean(l) for l in log_likelihoods])


class MaxProbabilityToken(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods'], 'token')

    def __str__(self):
        return 'MaxProbabilityToken'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats['greedy_log_likelihoods']
        return np.concatenate([-np.array(l[:-1]) for l in log_likelihoods])
