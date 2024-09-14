import numpy as np

from typing import List

from .ue_metric import UEMetric, normalize


class PredictionRejectionArea(UEMetric):
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __init__(self, max_rejection: float = 1.0):
        """
        Parameters:
            max_rejection (float): a maximum proportion of instances that will be rejected.
                1.0 indicates entire set, 0.5 - half of the set
        """
        super().__init__()
        self.max_rejection = max_rejection

    def __str__(self):
        if self.max_rejection == 1:
            return "prr"
        return f"prr_{self.max_rejection}"

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
        num_rej = int(self.max_rejection * num_obs)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)[-num_rej:]
        scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
        prr_score = np.sum(scores) / num_rej
        return prr_score
