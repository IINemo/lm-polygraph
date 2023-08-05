import numpy as np

from typing import Dict

from .estimator import Estimator


class MonteCarloNormalizedSequenceEntropy(Estimator):
    def __init__(self):
        super().__init__(['sample_log_probs', 'sample_tokens'], 'sequence')

    def __str__(self):
        return 'MonteCarloNormalizedSequenceEntropy'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats['sample_log_probs']
        tokens = stats['sample_tokens']
        return np.array([-np.mean(lp) / len(t) for lp, t in zip(logprobs, tokens)])
