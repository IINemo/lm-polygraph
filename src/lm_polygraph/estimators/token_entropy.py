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
    

class CumulativeMeanTokenEntropy(Estimator):
    """
    Estimates the cumulative average token-level entropy at each token step.
    It is calculated as the cumulative sum of token entropies divided by
    the number of tokens in the prefix. This shows how the average
    entropy evolves as the sequence is generated.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(["entropy"], "sequence")

    def __str__(self):
        return "CumulativeMeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the cumulative average token entropy at each token step
        for each sample.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which includes:
                * H(p(y_i | y_<i, x)) in 'entropy'
        Returns:
            np.ndarray: An object array (shape (N,)) where each element is
                        a 1D numpy array representing the cumulative
                        average entropy at each token step.
        """
        entropy = stats["entropy"]
        
        cumulative_entropy = [
            np.cumsum(e) / np.arange(1, len(e) + 1) if len(e) > 0 else np.array([])
            for e in entropy
        ]
        
        return np.array(cumulative_entropy, dtype=object)

    @property
    def returns_cumulative(self) -> bool:
        return True


class SortedCumulativeMeanTokenEntropy(Estimator):
    """
    Estimates the cumulative average token-level entropy by first sorting the entropies.
    It is calculated as the cumulative sum of token entropies divided by
    the number of tokens in the prefix. 
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(["entropy"], "sequence")

    def __str__(self):
        return "SortedCumulativeMeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the cumulative average token entropy by first sorting the entropies
        for each sample.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which includes:
                * H(p(y_i | y_<i, x)) in 'entropy'
        Returns:
            np.ndarray: An object array (shape (N,)) where each element is
                        a 1D numpy array representing the cumulative
                        average entropy at each token step.
        """
        entropy = [np.sort(e) for e in stats["entropy"]]
        
        
        cumulative_entropy = [
            np.cumsum(e) / np.arange(1, len(e) + 1) if len(e) > 0 else np.array([])
            for e in entropy
        ]
        
        return np.array(cumulative_entropy, dtype=object)

    @property
    def returns_cumulative(self) -> bool:
        return True


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
        return [np.array(e[:-1]) for e in entropy]
