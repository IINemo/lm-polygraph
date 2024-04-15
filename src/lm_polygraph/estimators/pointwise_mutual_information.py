import numpy as np

from typing import Dict

from .estimator import Estimator


class MeanPointwiseMutualInformation(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model using Pointwise Mutual Information.
    The sequence-level estimation is calculated as average token-level PMI estimations.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(
            ["greedy_log_likelihoods", "greedy_lm_log_likelihoods"], "sequence"
        )

    def __str__(self):
        return "MeanPointwiseMutualInformation"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the mean CPMI uncertainties with minus sign for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * log p(y_i | y_<i) in 'greedy_lm_log_likelihoods'.
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["greedy_log_likelihoods"]
        lm_logprobs = stats["greedy_lm_log_likelihoods"]
        mi_scores = []
        for lp, lm_lp in zip(logprobs, lm_logprobs):
            mi_scores.append([])
            for t in range(len(lp)):
                mi_scores[-1].append(lp[t] - (lm_lp[t] if t > 0 else 0))
        return np.array([-np.mean(sc) for sc in mi_scores])


class PointwiseMutualInformation(Estimator):
    """
    Estimates the token-level uncertainty of a language model using Pointwise Mutual Information.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(
            ["greedy_log_likelihoods", "greedy_lm_log_likelihoods"], "token"
        )

    def __str__(self):
        return "PointwiseMutualInformation"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the PMI uncertainties with minus sign for each token in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * p(y_i | y_<i) in 'greedy_lm_log_likelihoods'.
        Returns:
            np.ndarray: concatenated float uncertainty for each token in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["greedy_log_likelihoods"]
        lm_logprobs = stats["greedy_lm_log_likelihoods"]
        mi_scores = []
        for lp, lm_lp in zip(logprobs, lm_logprobs):
            mi_scores.append([])
            for t in range(len(lp)):
                mi_scores[-1].append(lp[t] - (lm_lp[t] if t > 0 else 0))
        return np.concatenate([-np.array(sc[:-1]) for sc in mi_scores])
