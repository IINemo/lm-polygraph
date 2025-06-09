"""Token entropy-based uncertainty estimation for language models."""

import numpy as np
from typing import Dict
from .estimator import Estimator


class MeanTokenEntropy(Estimator):
    """
    Mean token entropy uncertainty estimator for language models.
    
    Calculates the average entropy across all tokens in generated sequences
    as a measure of uncertainty. Entropy quantifies the uncertainty in the
    model's predictions at each token position, with higher entropy indicating
    more uniform probability distributions over the vocabulary.
    
    This estimator is useful for:
    - Identifying sequences where the model is consistently uncertain
    - Comparing overall uncertainty levels between different generations
    - Detecting hallucinations that arise from high prediction uncertainty
    
    The method computes Shannon entropy for each token's probability distribution
    and returns the mean across all tokens in the sequence.
    
    Attributes:
        dependencies (List[str]): Requires 'greedy_token_entropy' statistics
        level (str): Operates at 'sequence' level
    
    References:
        Fomicheva et al., 2020. "Unsupervised Quality Estimation for Neural Machine
        Translation" (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/)
    
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import MeanTokenEntropy
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = MeanTokenEntropy()
        >>> uncertainty = estimate_uncertainty(model, estimator, "Define entropy")
        >>> print(f"Mean token entropy: {uncertainty.uncertainty}")
        
    See Also:
        TokenEntropy: For token-level entropy values
        MaxTokenEntropy: For maximum entropy across tokens
        Perplexity: Alternative sequence-level uncertainty measure
    """
    
    def __init__(self):
        """Initialize MeanTokenEntropy estimator with required dependencies."""
        super().__init__(["greedy_token_entropy"], "sequence")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate mean token entropy for each sequence.
        
        Computes the average entropy across all tokens in each generated
        sequence. Higher values indicate that the model was more uncertain
        on average when generating the sequence.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'greedy_token_entropy': List of arrays with entropy values
                  for each token in each sequence. Each array has shape
                  (n_tokens,) with entropy values typically in range [0, 5+]
                   
        Returns:
            np.ndarray: Array of mean entropy values for each sequence.
                Shape: (n_sequences,). Higher values indicate higher average
                uncertainty across tokens. Typical range: [0, 3] depending
                on model and generation.
            
        Raises:
            KeyError: If 'greedy_token_entropy' is not in stats
        """
        token_entropies = stats["greedy_token_entropy"]
        mean_entropies = []
        for ent in token_entropies:
            if len(ent) == 0:
                ue = 0
            else:
                ue = np.mean(ent)
            mean_entropies.append(ue)
        return np.array(mean_entropies)


class TokenEntropy(Estimator):
    """
    Token-level entropy uncertainty estimator for language models.
    
    Calculates the entropy for each token in generated sequences as a measure
    of uncertainty. Unlike MeanTokenEntropy, this estimator returns uncertainty
    values for each individual token, allowing fine-grained analysis of where
    the model is most uncertain within a generation.
    
    Token entropy is particularly useful for:
    - Identifying specific positions where the model is uncertain
    - Highlighting potentially unreliable parts of generated text
    - Creating uncertainty heatmaps over generated sequences
    - Token-level hallucination detection
    
    The method computes Shannon entropy H(p) = -Σ p(x) log p(x) for each
    token's probability distribution over the vocabulary.
    
    Attributes:
        dependencies (List[str]): Requires 'greedy_token_entropy' statistics
        level (str): Operates at 'token' level
    
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import TokenEntropy
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = TokenEntropy()
        >>> # Get token-level uncertainties
        >>> result = estimate_uncertainty(model, estimator, "Explain thermodynamics")
        >>> # result.uncertainty will contain entropy for each token
        
    See Also:
        MeanTokenEntropy: For sequence-level average entropy
        MaxTokenEntropy: For maximum entropy in sequence
        AttentionScore: Alternative token-level uncertainty measure
    """
    
    def __init__(self):
        """Initialize TokenEntropy estimator with required dependencies."""
        super().__init__(["greedy_token_entropy"], "token")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "TokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Return token-level entropy values for all sequences.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'greedy_token_entropy': List of arrays with entropy values
                  for each token in each sequence
                   
        Returns:
            np.ndarray: Concatenated array of entropy values for all tokens
                across all sequences. Shape: (total_tokens,). Values typically
                range from 0 (certain) to 5+ (very uncertain) depending on
                vocabulary size and model confidence.
            
        Raises:
            KeyError: If 'greedy_token_entropy' is not in stats
            
        Note:
            The returned array concatenates token entropies from all sequences.
            To map back to specific sequences, you need to track sequence lengths.
        """
        token_entropies = stats["greedy_token_entropy"]
        return np.concatenate(token_entropies)
