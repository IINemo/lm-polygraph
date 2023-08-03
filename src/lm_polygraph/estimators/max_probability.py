import numpy as np

from typing import Dict

from .estimator import Estimator


class MaximumSequenceProbability(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods'], 'sequence')

    def __str__(self):
        return 'MaximumSequenceProbability'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats['greedy_log_likelihoods']
        return np.array([-np.sum(l) for l in log_likelihoods])


class MaximumTokenProbability(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods'], 'token')

    def __str__(self):
        return 'MaximumTokenProbability'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats['greedy_log_likelihoods']
        return np.concatenate([-np.exp(np.array(l[:-1])) for l in log_likelihoods])
