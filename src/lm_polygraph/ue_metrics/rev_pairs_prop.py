import numpy as np

from typing import List

from .ue_metric import UEMetric


class ReversedPairsProportion(UEMetric):
    """
    Calculates Reversed Pairs Proportion metrics.
    For uncetainty estimations `e` and ground-truth uncertainties `g`,
    the class calculates the proportion of pairs (i, j), such that
    e[i] < e[j] and g[i] > g[j].
    """

    def __str__(self):
        return "rpp"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Measures the proportion of reversed pairs between `estimator` and `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: proportion of reversed pairs.
                Lower values indicate better uncertainty estimations.
        """
        confidence = -np.array(estimator)
        target = np.array(target)
        cnts = []
        for i in range(len(confidence)):
            # greater confidences = lower uncertainties
            cnts.append(
                np.logical_and(confidence > confidence[i], target < target[i]).mean()
            )
        return np.mean(cnts).item()
