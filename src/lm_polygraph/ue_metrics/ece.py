import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler

from .ue_metric import UEMetric


class ECE(UEMetric):
    """
    Expected Calibration Error (ECE) metric. Only applicable to binary quality metrics.

    Without normalization, estimates must be negative confidences in [-1, 0].
    Set normalize=True to min-max scale arbitrary finite uncertainty estimates.
    """

    def __init__(self, normalize=False, n_bins=20):
        super().__init__()
        self.normalize = normalize
        self.n_bins = n_bins

    def __str__(self):
        return "ece"

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Performs min-max normalization of scores.
        Parameters:
            scores (List[float]): List of scores to normalize.
        Returns:
            List[float]: Normalized scores.
        """

        scores = np.asarray(scores).reshape(-1, 1)

        return MinMaxScaler().fit_transform(scores).flatten()

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")
        if len(estimator) == 0:
            raise ValueError("Estimator and target must not be empty.")
        estimator = np.asarray(estimator)
        target = np.asarray(target)
        if not np.all(np.isfinite(estimator)):
            raise ValueError("Estimator values must be finite.")

        # ECE expects confidence, not uncertainty, so we invert the estimator
        confidences = -estimator

        if self.normalize:
            confidences = self.normalize_scores(confidences)
        elif np.any((confidences < 0) | (confidences > 1)):
            raise ValueError(
                "Confidences (-estimator) must be in [0, 1]. "
                "Use normalize=True to min-max scale uncertainty estimates."
            )

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece, N = 0.0, len(confidences)
        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (
                (confidences > lo) & (confidences <= hi)
                if i > 0
                else (confidences >= lo) & (confidences <= hi)
            )
            if not np.any(in_bin):
                continue
            acc_bin = np.mean(target[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / N) * abs(acc_bin - conf_bin)

        return ece
