import numpy as np

from typing import Dict

from .estimator import Estimator


class MaximumSequenceProbability(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model by calculating the
    log-probability of the generation with minus sign.
    It is calculated as the sum of log-probabilities in each token.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MaximumSequenceProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the minus log-probability of each sample in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
        Returns:
            np.ndarray: minus log probabilities for each sample.
                Higher values indicate more uncertain samples.
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.sum(log_likelihood) for log_likelihood in log_likelihoods])


class MaximumTokenProbability(Estimator):
    """
    Estimates the token-level uncertainty of a language model by calculating the
    log-probability for each token during autoregressive generation.
    """

    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "token")

    def __str__(self):
        return "MaximumTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the minus log-probability of each token in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
        Returns:
            np.ndarray: concatenated minus log probabilities for each token.
                Higher values indicate more uncertain samples.
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.concatenate(
            [
                -np.exp(np.array(log_likelihood[:-1]))
                for log_likelihood in log_likelihoods
            ]
        )
