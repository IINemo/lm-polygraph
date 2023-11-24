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
        raise Exception("Not implemented")

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
        raise Exception("Not implemented")


def get_random_scores(function, metrics, num_iter=1000, seed=42):
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    value = []
    for i in range(num_iter):
        np.random.shuffle(rand_scores)
        rand_val = function(rand_scores, metrics)
        value.append(rand_val)
    return np.mean(value)


def normalize_metric(target_score, oracle_score, random_score):
    if not (oracle_score == random_score):
        target_score = (target_score - random_score) / (oracle_score - random_score)
    return target_score
