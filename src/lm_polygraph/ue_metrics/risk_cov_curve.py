import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize


class RiskCoverageCurveAUC(UEMetric):
    def __str__(self, normalize=True):
        self.normalize = normalize
        return "rcc-auc"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        target = normalize(target)
        # greater is better
        risk = 1 - np.array(target)
        cr_pair = list(zip(estimator, risk))
        cr_pair.sort(key=lambda x: x[0], reverse=True)
        cumulative_risk = np.cumsum([x[1] for x in cr_pair])
        if self.normalize:
            cumulative_risk = cumulative_risk / np.arange(1, len(estimator) + 1)
        return cumulative_risk.mean()
