import numpy as np
from sklearn.metrics import roc_auc_score

from typing import List

from .ue_metric import UEMetric, skip_target_nans


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
        t, e = skip_target_nans(target, estimator)
        return roc_auc_score(t, e)


class AUROC(UEMetric):
    """
    It's the same, but for selective prediction.
    """

    def __init__(self, binarize_metric: bool = False, metric_threshold: float = 0.5):
        self.binarize_metric = binarize_metric
        self.metric_threshold = metric_threshold

    def __str__(self):
        return "auroc"

    def preprocess_inf(self, x, array):
        if not np.isinf(x):
            return x
        elif x > 0:
            return array.max() + 1
        else:
            return array.min() - 1

    def __call__(self, estimator: List[float], target: List[int]) -> float:
        estimator = [self.preprocess_inf(x, estimator) for x in estimator]
        t, e = skip_target_nans(target, estimator)

        # Invert the estimator values, because higher values should indicate higher accuracy
        e = -np.array(e)

        if self.binarize_metric:
            e = (e > self.metric_threshold).astype(int)

        return roc_auc_score(t, e)
