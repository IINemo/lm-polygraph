import numpy as np

from typing import Dict

from .estimator import Estimator


class MeanConditionalPointwiseMutualInformation(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    Conditional Pointwise Mutual Information (CPMI) as provided in the paper https://arxiv.org/abs/2210.13210.
    The sequence-level estimation is calculated as average token-level CPMI estimations.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self, tau: float = 0.0656, lambd: float = 3.599):
        """
        Parameters:
            tau (float): the threshold parameter for entropy
                (default: 0.0656 - best according to https://arxiv.org/abs/2210.13210)
            lambd (float): scale factor for pointwise mutual information
                (default: 3.599 - best according to https://arxiv.org/abs/2210.13210)
        """
        super().__init__(
            ["greedy_log_likelihoods", "greedy_lm_log_likelihoods", "entropy"],
            "sequence",
        )
        self.tau = tau
        self.lambd = lambd

    def __str__(self):
        return "MeanConditionalPointwiseMutualInformation"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the mean CPMI uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * log p(y_i | y_<i) in 'greedy_lm_log_likelihoods',
                * Entropy(* | y_<i, x) in 'entropy'.
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["greedy_log_likelihoods"]
        lm_logprobs = stats["greedy_lm_log_likelihoods"]
        entropies = stats["entropy"]
        mi_scores = []
        for lp, lm_lp, ent in zip(logprobs, lm_logprobs, entropies):
            mi_scores.append([])
            for t in range(len(lp)):
                score = lp[t]
                if t > 0 and ent[t] >= self.tau:
                    score -= self.lambd * lm_lp[t]
                mi_scores[-1].append(score)
        return np.array([-np.mean(sc) for sc in mi_scores])


class ConditionalPointwiseMutualInformation(Estimator):
    """
    Estimates the token-level uncertainty of a language model following the method of
    Conditional Pointwise Mutual Information (CPMI) as provided in the paper https://arxiv.org/abs/2210.13210.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self, tau: float = 0.0656, lambd: float = 3.599):
        """
        Parameters:
            tau (float): the threshold parameter for entropy
                (default: 0.0656 - best according to https://arxiv.org/abs/2210.13210)
            lambd (float): scale factor for pointwise mutual information
                (default: 3.599 - best according to https://arxiv.org/abs/2210.13210)
        """
        super().__init__(
            ["greedy_log_likelihoods", "greedy_lm_log_likelihoods", "entropy"], "token"
        )
        self.tau = tau
        self.lambd = lambd

    def __str__(self):
        return "ConditionalPointwiseMutualInformation"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the CPMI uncertainties for each token in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * p(y_i | y_<i) in 'greedy_lm_log_likelihoods',
                * Entropy(* | y_<i, x) in 'entropy'.
        Returns:
            np.ndarray: concatenated float uncertainty for each token in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["greedy_log_likelihoods"]
        lm_logprobs = stats["greedy_lm_log_likelihoods"]
        entropies = stats["entropy"]
        mi_scores = []
        for lp, lm_lp, ent in zip(logprobs, lm_logprobs, entropies):
            mi_scores.append([])
            for t in range(len(lp)):
                score = lp[t]
                if t > 0 and ent[t] >= self.tau:
                    score -= self.lambd * lm_lp[t]
                mi_scores[-1].append(score)
        return np.concatenate([-np.array(sc[:-1]) for sc in mi_scores])
