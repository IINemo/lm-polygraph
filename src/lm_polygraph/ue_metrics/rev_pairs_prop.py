import numpy as np

from typing import List

from .ue_metric import UEMetric


class ReversedPairsProportion(UEMetric):
    def __str__(self):
        return 'rpp'

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        # smaller is better
        ue = -np.array(estimator)
        target = np.array(target)
        cnts = []
        for i in range(len(ue)):
            cnts.append(np.logical_and(ue > ue[i], target < target[i]).mean())
        return np.mean(cnts).item()
