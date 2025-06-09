"""Maximum probability-based uncertainty estimation for language models."""

import numpy as np
from typing import Dict
from .estimator import Estimator


class MaximumSequenceProbability(Estimator):
    """
    Maximum sequence probability uncertainty estimator.
    
    Estimates uncertainty as the negative joint probability of the entire
    generated sequence. This method considers the cumulative probability
    of generating the exact sequence, with lower probabilities indicating
    higher uncertainty.
    
    The estimator is based on the principle that sequences the model assigns
    high probability to are more reliable. It computes the joint probability
    as the sum of log probabilities of all tokens in the sequence.
    
    This estimator is useful for:
    - Ranking multiple generated sequences by confidence
    - Identifying unlikely or potentially hallucinated sequences
    - Comparing relative confidence across different generations
    
    Attributes:
        dependencies (List[str]): Requires 'greedy_log_likelihoods' statistics
        level (str): Operates at 'sequence' level
        
    References:
        Malinin & Gales, 2021. "Uncertainty Estimation in Autoregressive 
        Structured Prediction" (https://openreview.net/forum?id=jN5y-zb5Q7m)
    
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import MaximumSequenceProbability
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = MaximumSequenceProbability()
        >>> uncertainty = estimate_uncertainty(model, estimator, "What is 2+2?")
        >>> print(f"Sequence probability score: {uncertainty.uncertainty}")
        
    See Also:
        MaximumTokenProbability: For maximum probability at token level
        Perplexity: Alternative probability-based measure
    """
    
    def __init__(self):
        """Initialize MaximumSequenceProbability with required dependencies."""
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "MaximumSequenceProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate negative sequence probability as uncertainty.
        
        Computes the joint probability of the entire sequence by summing
        log probabilities of all tokens, then negates it to create an
        uncertainty score where higher values indicate higher uncertainty.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'greedy_log_likelihoods': List of arrays with log probabilities
                  for each token in each sequence
                   
        Returns:
            np.ndarray: Array of negative log probabilities (uncertainties) for
                each sequence. Shape: (n_sequences,). Higher values indicate
                lower sequence probability and thus higher uncertainty.
                Typical range: [0, 100+] depending on sequence length.
            
        Raises:
            KeyError: If 'greedy_log_likelihoods' is not in stats
            
        Note:
            The uncertainty grows with sequence length since it's a joint
            probability. For length-normalized scores, consider using Perplexity.
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.sum(ll) for ll in log_likelihoods])


class MaximumTokenProbability(Estimator):
    """
    Maximum token probability uncertainty estimator.
    
    Estimates token-level uncertainty as the negative log probability of each
    token. This provides a fine-grained view of model confidence at each
    position in the generated sequence.
    
    Unlike sequence-level estimators, this method returns uncertainty values
    for individual tokens, allowing identification of specific positions where
    the model was less confident during generation.
    
    This estimator is useful for:
    - Creating token-level confidence visualizations
    - Identifying specific uncertain tokens in a generation
    - Token-level hallucination detection
    - Fine-grained analysis of model behavior
    
    Attributes:
        dependencies (List[str]): Requires 'greedy_log_likelihoods' statistics
        level (str): Operates at 'token' level
    
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import MaximumTokenProbability
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = MaximumTokenProbability()
        >>> result = estimate_uncertainty(model, estimator, "Explain gravity")
        >>> # result.uncertainty contains probability for each token
        
    See Also:
        MaximumSequenceProbability: For sequence-level probability
        TokenEntropy: Alternative token-level uncertainty measure
    """
    
    def __init__(self):
        """Initialize MaximumTokenProbability with required dependencies."""
        super().__init__(["greedy_log_likelihoods"], "token")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "MaximumTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate negative log probability for each token.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'greedy_log_likelihoods': List of arrays with log probabilities
                  for each token in each sequence
                   
        Returns:
            np.ndarray: Concatenated array of negative log probabilities for all
                tokens across all sequences. Shape: (total_tokens,). Higher values
                indicate lower token probability and thus higher uncertainty.
                Typical range: [0, 10+] per token.
            
        Raises:
            KeyError: If 'greedy_log_likelihoods' is not in stats
            
        Note:
            The returned array concatenates token probabilities from all sequences.
            To map back to specific sequences, you need to track sequence lengths.
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.concatenate([-ll for ll in log_likelihoods])
