import numpy as np
from typing import List

from .ue_metric import UEMetric
from lm_polygraph.normalizers.isotonic_pcc import IsotonicPCCNormalizer


class IsotonicPCC(UEMetric):
    """
    Perform leave-one-out isotonic regression calibration and measure mean squared error (MSE).
    For each point, fit IsotonicPCCNormalizer on all other points, then predict and compare with metric.
    """

    def __str__(self):
        return "isotonic-pcc"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")
        estimator = np.asarray(estimator)
        target = np.asarray(target)
        n = len(target)
        preds = np.zeros(n)
        # Leave-one-out calibration and prediction
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            normalizer = IsotonicPCCNormalizer()
            normalizer.fit(target[mask], estimator[mask])
            preds[i] = normalizer.transform(np.array([estimator[i]]))[0]
        mse = np.mean((preds - target) ** 2)
        return mse
