import numpy as np
from scipy import stats
from typing import List

from .ue_metric import UEMetric, normalize


class KendallTauCorrelation(UEMetric):
    """
    Calculates the Kendall’s tau correlation.
    """

    def __str__(self):
        return "kendalltau"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Calculates the Kendall’s tau, a correlation measure between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: Kendall’s tau correlation
                Higher values indicate better uncertainty estimations.
        """
        target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)

        return stats.kendalltau(ue, -target).correlation
