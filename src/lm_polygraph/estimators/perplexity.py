import numpy as np

from typing import Dict

from .estimator import Estimator


class Perplexity(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Perplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return [ -np.cumsum(ll) / np.arange(1, len(ll) + 1)  for ll in log_likelihoods ] # np.array([-np.mean(ll) for ll in log_likelihoods])
