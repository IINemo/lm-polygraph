import numpy as np

from typing import Dict

from .estimator import Estimator


class MeanPointwiseMutualInformation(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods', 'greedy_lm_log_likelihoods'], 'sequence')

    def __str__(self):
        return 'MeanPointwiseMutualInformation'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats['greedy_log_likelihoods']
        lm_logprobs = stats['greedy_lm_log_likelihoods']
        mi_scores = []
        for lp, lm_lp in zip(logprobs, lm_logprobs):
            mi_scores.append([])
            for t in range(len(lp)):
                mi_scores[-1].append(lp[t] - (lm_lp[t - 1] if t > 0 else 0))
        return np.array([-np.mean(sc) for sc in mi_scores])


class PointwiseMutualInformation(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods', 'greedy_lm_log_likelihoods'], 'token')

    def __str__(self):
        return 'PointwiseMutualInformation'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats['greedy_log_likelihoods']
        lm_logprobs = stats['greedy_lm_log_likelihoods']
        mi_scores = []
        for lp, lm_lp in zip(logprobs, lm_logprobs):
            mi_scores.append([])
            for t in range(len(lp)):
                mi_scores[-1].append(lp[t] - (lm_lp[t - 1] if t > 0 else 0))
        return np.concatenate([-np.array(sc[:-1]) for sc in mi_scores])
