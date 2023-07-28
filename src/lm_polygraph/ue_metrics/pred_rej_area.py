import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize


class PredictionRejectionArea(UEMetric):
    def __str__(self):
        return "prr"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        target = normalize(target)
        # estimator: greater is less certain
        # ue: greater is more certain
        ue = -np.array(estimator)
        num_obs = len(ue)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)
        scores = (cumsum / np.arange(1, num_obs + 1))[::-1]
        prr_score = np.sum(scores) / num_obs
        scores = np.append(scores, 1)
        return prr_score
