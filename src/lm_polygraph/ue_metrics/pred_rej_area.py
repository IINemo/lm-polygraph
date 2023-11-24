import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize


class PredictionRejectionArea(UEMetric):
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __str__(self):
        return "prr"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Measures the area under the Prediction-Rejection curve between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Prediction-Rejection curve.
                Higher values indicate better uncertainty estimations.
        """
        target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)
        num_obs = len(ue)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)
        scores = (cumsum / np.arange(1, num_obs + 1))[::-1]
        prr_score = np.sum(scores) / num_obs
        return prr_score
