import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize


class RiskCoverageCurveAUC(UEMetric):
    """
    Calculates area under the Risk-Coverage curve.
    """

    def __init__(self, normalize: bool = True):
        """
        Parameters:
            normalize (bool): whether the risk curve should be normalized to 0..1
        """
        self.normalize = normalize

    def __str__(self):
        return "rcc-auc"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Measures the area under the Risk-Coverage curve between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Risk-Coverage curve.
                Lower values indicate better uncertainty estimations.
        """
        target = normalize(target)
        risk = 1 - np.array(target)
        cr_pair = list(zip(estimator, risk))
        cr_pair.sort(key=lambda x: x[0])
        cumulative_risk = np.cumsum([x[1] for x in cr_pair])
        if self.normalize:
            cumulative_risk = cumulative_risk / np.arange(1, len(estimator) + 1)
        return cumulative_risk.mean()
