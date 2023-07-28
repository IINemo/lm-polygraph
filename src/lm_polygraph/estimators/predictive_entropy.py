import numpy as np

from typing import Dict

from .estimator import Estimator


class PredictiveEntropy(Estimator):
    def __init__(self):
        super().__init__(["sample_log_probs"], "sequence")

    def __str__(self):
        return "PredictiveEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats["sample_log_probs"]
        return np.array([-np.mean(lp) for lp in logprobs])
