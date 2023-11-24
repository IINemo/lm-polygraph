import numpy as np
from scipy import stats
from typing import List

from .ue_metric import UEMetric, normalize


class SpearmanRankCorrelation(UEMetric):
    """
    Calculates the Spearman's rank correlation coefficient.
    """

    def __str__(self):
        return "spearmanr"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Calculates the Spearman's rank correlation coefficient, a correlation measure between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: Spearman's rank correlation coefficient
                Higher values indicate better uncertainty estimations.
        """
        target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)

        return stats.spearmanr(ue, -target).correlation
