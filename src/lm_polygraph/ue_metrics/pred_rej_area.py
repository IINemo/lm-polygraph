import numpy as np
from sklearn.metrics import auc

from typing import List

from .ue_metric import UEMetric, normalize

NUM_RANDOM_SAMPLES=100

class PredictionRejectionArea(UEMetric):
    """
    Calculates area under Prediction-Rejection curve.
    """

    def __str__(self):
        return 'prr'

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
        num_obs = len(target)
        cum_obs = np.arange(1, num_obs + 1)

        #target = normalize(target)
        # ue: greater is more uncertain
        ue = np.array(estimator)
        # Sort in ascending order: the least uncertain come first
        ue_argsort = np.argsort(ue)
        # want sorted_metrics to be increasing => smaller scores is better
        sorted_metrics = np.array(target)[ue_argsort]
        # Since we want all plots to coincide when all the data is discarded
        cumsum = np.cumsum(sorted_metrics)
        scores = (cumsum / cum_obs)
        estimator_area = auc(cum_obs, scores)

        random_areas = []
        for _ in range(NUM_RANDOM_SAMPLES):
            random_scores = np.cumsum(np.random.permutation(target)[::-1]) / cum_obs
            random_areas.append(auc(cum_obs, random_scores))
        random_area = np.mean(random_areas)

        oracle_scores = np.cumsum(np.sort(target)[::-1]) / cum_obs
        oracle_area = auc(cum_obs, oracle_scores)

        num = (estimator_area - random_area)
        denom = (oracle_area - random_area)

        if denom == 0:
            return 0

        prr = num / denom

        return prr
