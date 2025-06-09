"""Reflexive uncertainty estimation using model self-evaluation."""

import numpy as np
from typing import Dict
from .estimator import Estimator


class PTrue(Estimator):
    """
    P(True) reflexive uncertainty estimator.
    
    This estimator asks the model to evaluate the truthfulness of its own
    generated statements, leveraging the model's ability to assess confidence
    in its outputs. It appends a question about truthfulness and analyzes
    the probability the model assigns to "True" vs "False" responses.
    
    The method is based on the observation that language models can often
    judge the correctness of statements, even their own. By asking the model
    "Is the above claim true or false?", we can use the probability it
    assigns to "True" as a confidence measure.
    
    This estimator is useful for:
    - Models that have been trained with truthfulness objectives
    - Factual question answering scenarios  
    - Cases where the model has metalinguistic capabilities
    - Quick confidence estimation without multiple samples
    
    Attributes:
        dependencies (List[str]): Requires 'p_true' statistics
        level (str): Operates at 'sequence' level
        
    References:
        Kadavath et al., 2022. "Language Models (Mostly) Know What They Know"
        (https://arxiv.org/abs/2207.05221)
        
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import PTrue
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = PTrue()
        >>> result = estimate_uncertainty(
        ...     model, estimator,
        ...     "What year did World War II end?"
        ... )
        >>> # Lower uncertainty means model thinks its answer is true
        
    See Also:
        PTrueSampling: Variant using multiple samples
        Verbalized1S: Alternative reflexive method using verbal confidence
    
    Note:
        This method assumes the model understands the concept of truth/falsehood
        and can meaningfully evaluate its own outputs. Performance varies
        significantly across different models and domains.
    """
    
    def __init__(self):
        """Initialize PTrue estimator with required dependencies."""
        super().__init__(["p_true"], "sequence")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "PTrue"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty as negative probability of "True" response.
        
        The model is asked whether its generation is true or false, and we
        use the probability it assigns to "True" as a confidence measure.
        Higher P(True) indicates higher confidence, so we negate it to
        create an uncertainty score.
        
        Parameters:
            stats: Dictionary containing required statistics:
                - 'p_true': Array of probabilities that the model assigns
                  to the "True" token when asked about truthfulness.
                  Shape: (n_sequences,), values in [0, 1]
                   
        Returns:
            np.ndarray: Array of uncertainty scores (negative P(True)) for
                each sequence. Shape: (n_sequences,). Higher values indicate
                the model thinks its answer is more likely to be false.
                Range: [-1, 0] where -1 is maximum uncertainty.
            
        Raises:
            KeyError: If 'p_true' is not in stats
            
        Note:
            The quality of this estimate depends heavily on the model's
            calibration and its ability to assess its own outputs accurately.
        """
        p_true = stats["p_true"]
        return -p_true
