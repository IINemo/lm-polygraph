import numpy as np

from typing import List
from abc import ABC, abstractmethod


def normalize(target: List[float]):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


class UEMetric(ABC):
    """
    Abstract class, which measures the quality of uncertainty estimations from some Estimator using
    ground-truth uncertainty estimations calculated from some GenerationMetric.
    """

    @abstractmethod
    def __str__(self):
        """
        Abstract method. Returns unique name of the UEMetric.
        Class parameters which affect generation metric estimates should also be included in the unique name
        to diversify between UEMetric's.
        """
        raise Exception('Not implemented')

    @abstractmethod
    def __call__(self, estimator: List[float], target: List[float]) -> float:
        """
        Abstract method. Measures the quality of uncertainty estimations `estimator`
        by comparing them to the ground-truth uncertainty estimations `target`.

        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: a quality measure of `estimator` estimations.
                Higher values can indicate either better or lower qualities,
                which depends on a particular implementation.
        """
        raise Exception('Not implemented')
