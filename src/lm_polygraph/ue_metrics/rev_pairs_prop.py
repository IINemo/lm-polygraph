import numpy as np

from typing import List

from .ue_metric import UEMetric


class ReversedPairsProportion(UEMetric):
    def __str__(self):
        return 'rpp'

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        # smaller is better
        confidence = -np.array(estimator)
        target = np.array(target)
        cnts = []
        for i in range(len(confidence)):
            cnts.append(np.logical_and(confidence > confidence[i], target < target[i]).mean())
        return np.mean(cnts).item()
