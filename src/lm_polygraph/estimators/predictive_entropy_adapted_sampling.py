import numpy as np

from typing import Dict

from .estimator import Estimator


class PredictiveEntropyAdaptedSampling(Estimator):
    def __init__(self):
        super().__init__(
            ["adapted_sample_log_probs", "adapted_sample_log_probs_gen"], "sequence"
        )

    def __str__(self):
        return "PredictiveEntropyAdaptedSampling"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_p = stats["adapted_sample_log_probs"]
        log_p_gen = stats["adapted_sample_log_probs_gen"]
        return np.array(
            [
                -(np.exp(np.array(log_p) - np.array(log_p_gen)) * log_p).mean()
                for log_p, log_p_gen in zip(log_p, log_p_gen)
            ]
        )  # Importance sampling
