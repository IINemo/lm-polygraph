import numpy as np

from typing import Dict

from .estimator import Estimator


class Perplexity(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Perplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.mean(ll) for ll in log_likelihoods])
    
class CumulativePerplexity(Estimator):
    """
    Estimates the cumulative sequence-level uncertainty at each token step
    by calculating the average negative log-likelihood (log-perplexity)
    for each prefix.
    
    This shows how log-perplexity evolves as the sequence is generated.
    Works only with whitebox models.
    """

    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "CumulativePerplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the cumulative average negative log-probability (log-perplexity)
        at each token step for each sample.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
        Returns:
            np.ndarray: An object array (shape (N,)) where each element is
                        a 1D numpy array representing the cumulative
                        log-perplexity at each token step.
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        
        cumulative_ppl = [
            -np.cumsum(ll) / np.arange(1, len(ll) + 1) if len(ll) > 0 else np.array([])
            for ll in log_likelihoods
        ]
        
        return np.array(cumulative_ppl, dtype=object)
    
    @property
    def returns_cumulative(self) -> bool:
        return True
