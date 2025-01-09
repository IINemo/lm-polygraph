import numpy as np

from typing import Dict

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class Perplexity(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Perplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.mean(ll) for ll in log_likelihoods])

class SampledPerplexity(Estimator):
    def __init__(self, sample_strategy: str = "first"):
        super().__init__(["sample_log_likelihoods"], "sequence")
        self.sample_strategy = sample_strategy

    def __str__(self):
        return sample_strategy_to_prefix(self.sample_strategy) + "SampledPerplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["sample_log_likelihoods"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        ppl = []
        for best_id, sample_log_likelihoods in zip(sample_ids, log_likelihoods):
            ppl.append(np.mean(sample_log_likelihoods[best_id]))

        return -np.array(ppl)
