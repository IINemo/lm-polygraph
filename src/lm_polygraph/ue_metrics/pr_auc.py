import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from typing import List

from .ue_metric import UEMetric, skip_target_nans


class PRAUC(UEMetric):
    def __init__(self, positive_class: int = 1, negative_class: int = 0):
        super().__init__()
        self.positive_class = positive_class
        self.negative_class = negative_class

    def __str__(self):
        return "pr-auc"

    def __call__(self, estimator: List[float], target: List[int]) -> float:
        # nans in the target might correspond to non-labeled claims
        t, e = skip_target_nans(target, estimator)
        if np.isinf(e).any():
            # Average precision depends only on ordering and ties. Finite ranks
            # preserve both, even for all-infinite scores or float extrema where
            # adding/subtracting one cannot produce a distinct finite sentinel.
            e = rankdata(e, method="dense")
        assert all(x in [self.positive_class, self.negative_class] for x in t)
        if self.positive_class < self.negative_class:
            # swap classes
            t = self.positive_class + self.negative_class - np.array(t)
            e = -np.array(e)
        return average_precision_score(t, e)
