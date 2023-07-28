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
    @abstractmethod
    def __str__(self):
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, estimator: List[float], target: List[float]) -> float:
        raise Exception("Not implemented")
