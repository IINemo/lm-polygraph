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
        return np.array([-np.mean(ll) for ll in log_likelihoods])

class SampledPerplexity(Estimator):
    def __init__(self):
        super().__init__(["sample_log_likelihoods"], "sequence")

    def __str__(self):
        return "SampledPerplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["sample_log_likelihoods"]
        ppl = [np.mean(sample_log_likelihoods[0]) for sample_log_likelihoods in log_likelihoods]
        return -np.array(ppl)
