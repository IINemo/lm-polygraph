import numpy as np

from typing import Dict

from .estimator import Estimator


class MaxProbabilityNormalizedSeq(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MaxProbabilityNormalizedSeq"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.mean(l) / len(l) for l in log_likelihoods])


class MaxProbabilityNormalizedToken(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "token")

    def __str__(self):
        return "MaxProbabilityNormalizedToken"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.concatenate(
            [-np.array(l[:-1]) / np.arange(1, len(l)) for l in log_likelihoods]
        )
