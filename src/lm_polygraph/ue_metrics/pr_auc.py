import numpy as np
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

    def preprocess_inf(self, x, array):
        if not np.isinf(x):
            return x
        elif x > 0:
            return array.max() + 1
        else:
            return array.min() - 1

    def __call__(self, estimator: List[float], target: List[int]) -> float:
        estimator = [self.preprocess_inf(x, estimator) for x in estimator]
        # nans in the target might correspond to non-labeled claims
        t, e = skip_target_nans(target, estimator)
        assert all(x in [self.positive_class, self.negative_class] for x in t)
        if self.positive_class < self.negative_class:
            # swap classes
            t = self.positive_class + self.negative_class - np.array(t)
            e = -np.array(e)
        return average_precision_score(t, e)
