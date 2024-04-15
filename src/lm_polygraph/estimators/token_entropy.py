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
