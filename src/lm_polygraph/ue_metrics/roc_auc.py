import numpy as np
from sklearn.metrics import roc_auc_score

from typing import List

from .ue_metric import UEMetric


class ROCAUC(UEMetric):
    is_ood_metric = True
    def __str__(self):
        return 'roc-auc'

    def __call__(self, estimator: List[float], target: List[int]) -> float:
        return roc_auc_score(target, estimator)