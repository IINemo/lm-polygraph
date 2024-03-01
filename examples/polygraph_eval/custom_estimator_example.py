import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


# This estimator is essentialy -MSP
class CustomEstimatorExample(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "CustomEstimatorExample"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([np.sum(logl) for logl in log_likelihoods])
