"""Perplexity-based uncertainty estimation for language models."""

import numpy as np

from typing import Dict

from .estimator import Estimator


class Perplexity(Estimator):
    """
    Perplexity-based uncertainty estimator for language models.
    
    Calculates the perplexity of generated sequences as a measure of uncertainty.
    Lower perplexity indicates higher confidence in the generation. Perplexity is
    computed as the exponential of the average negative log-likelihood of tokens.
    
    This estimator is particularly useful for:
    - Quick uncertainty estimation with low computational overhead
    - Comparing relative uncertainties across different inputs
    - Detecting potentially problematic or unlikely generations
    
    The method computes uncertainty as the negative mean of log probabilities,
    which represents the average surprisal per token in the generated sequence.
    
    Attributes:
        dependencies (List[str]): Requires 'greedy_log_likelihoods' statistics
        level (str): Operates at 'sequence' level
    
    References:
        Fomicheva et al., 2020. "Unsupervised Quality Estimation for Neural Machine
        Translation" (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/)
    
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import Perplexity
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = Perplexity()
        >>> uncertainty = estimate_uncertainty(model, estimator, "What is AI?")
        >>> print(f"Perplexity score: {uncertainty.uncertainty}")
        
    See Also:
        TokenEntropy: For token-level entropy estimation
        MaximumSequenceProbability: For probability-based uncertainty
    """
    
    def __init__(self):
        """Initialize the Perplexity estimator with required dependencies."""
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "Perplexity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate perplexity scores for the given statistics.
        
        Perplexity is calculated as the negative mean of log likelihoods,
        which represents the average uncertainty per token. Higher values
        indicate that the model found the generated sequence less likely.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'greedy_log_likelihoods': List of arrays with log probabilities
                  for each generated token in each sequence
                   
        Returns:
            np.ndarray: Array of perplexity scores (negative mean log likelihood)
                for each sequence. Shape: (n_sequences,). Higher values indicate 
                higher uncertainty. Typical range: [0, 10+] depending on model.
            
        Raises:
            KeyError: If 'greedy_log_likelihoods' is not in stats
            
        Note:
            The actual perplexity value would be exp(-mean(log_likelihoods)),
            but we return -mean(log_likelihoods) directly as it preserves
            the ordering and is more numerically stable.
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.mean(ll) for ll in log_likelihoods])
