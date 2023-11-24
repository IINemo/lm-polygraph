import numpy as np

from typing import Dict

from .estimator import Estimator


class PTrueSampling(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221 and model samples.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question `q`, model generation `a` and several text samples `s`,
    the method uses the following prompt:
        Question: {q}
        Here are some ideas that were brainstormed: {s}
        Possible answer: {a}
        Is the possible answer:
         (A) True
         (B) False
        The possible answer is:
    and calculates the log-probability of `True` token with minus sign.
    """

    def __init__(self):
        super().__init__(["p_true_sampling"], "sequence")

    def __str__(self):
        return "PTrueSampling"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates minus log-probability of true token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true_sampling'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        pue = stats["p_true_sampling"]
        return -np.array(pue)
