import numpy as np
from sklearn.metrics import roc_auc_score

from typing import List

from .ue_metric import UEMetric


class ROCAUC(UEMetric):
    is_ood_metric = True

    def __str__(self):
        return "roc-auc"

    def preprocess_inf(self, x, array):
        if not np.isinf(x):
            return x
        elif x > 0:
            return array.max() + 1
        else:
            return array.min() - 1

    def __call__(self, estimator: List[float], target: List[int]) -> float:
        estimator = [self.preprocess_inf(x, estimator) for x in estimator]
        return roc_auc_score(target, estimator)
