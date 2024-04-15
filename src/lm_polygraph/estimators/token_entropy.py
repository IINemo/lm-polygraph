import numpy as np

from typing import Dict

from .estimator import Estimator


class MeanTokenEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model by calculating the
    mean entropy among all tokens in the generation.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(["entropy"], "sequence")

    def __str__(self):
        return "MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the mean token entropy for each sample in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * Entropy(* | y_<i, x) in 'entropy'
        Returns:
            np.ndarray: minus log probabilities for each sample.
                Higher values indicate more uncertain samples.
        """
        entropy = stats["entropy"]
        return np.array([np.mean(e) for e in entropy])


class MaxTokenEntropyClaim(Estimator):
    """
    Estimates the claim-level maximum token entropy.
    """

    def __init__(self):
        super().__init__(["entropy", "claims"], "claim")

    def __str__(self):
        return "MaxTokenEntropyClaim"

    def _reduce(self, x: np.ndarray):
        return np.max(x)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the minus log-probability of each claim in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
                * list of extracted claims of type lm_polygraph.stat_calculators.extract_claims.Claim
                  in 'claims'
        Returns:
            np.ndarray: concatenated minus log probabilities for each claim.
                Higher values indicate more uncertain samples.
        """
        entropies = stats["entropy"]
        claims = stats["claims"]
        claim_ue = []
        for sample_ent, sample_claims in zip(entropies, claims):
            for claim in sample_claims:
                tokens = np.array(claim.aligned_tokens)
                claim_ent = np.array(sample_ent)[tokens]
                claim_ue.append(self._reduce(claim_ent))
        return np.array(claim_ue)


class TokenEntropy(Estimator):
    """
    Estimates the token-level uncertainty of a language model by calculating the
    entropy for each token in the generation.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(["entropy"], "token")

    def __str__(self):
        return "TokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the token entropy for each sample in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * Entropy(* | y_<i, x) in 'entropy'
        Returns:
            np.ndarray: concatenated entropies for each token.
                Higher values indicate more uncertain samples.
        """
        entropy = stats["entropy"]
        return np.concatenate([np.array(e[:-1]) for e in entropy])
