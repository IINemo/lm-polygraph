import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


# This estimator is essentialy MSP * 2
class CustomEstimatorExample(Estimator):
    def __init__(self):
        super().__init__(['greedy_log_likelihoods'], 'sequence')

    def __str__(self):
        return 'DoubleMaximumSequenceProbability'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats['greedy_log_likelihoods']
        return np.array([-2 * np.sum(l) for l in log_likelihoods])
