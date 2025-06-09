"""Label probability uncertainty estimation based on semantic clustering."""

import numpy as np

from typing import Dict

from .estimator import Estimator


class LabelProb(Estimator):
    """
    Label probability uncertainty estimator based on semantic clustering.
    
    This estimator measures uncertainty by analyzing the distribution of
    generated samples across semantic clusters. Higher concentration in
    a single cluster indicates lower uncertainty, while uniform distribution
    across multiple clusters suggests high uncertainty.
    
    The method works by:
    1. Generating multiple samples for the same input (done by stat calculator)
    2. Clustering samples by semantic similarity using NLI models
    3. Computing uncertainty as 1 - (largest_cluster_size / total_samples)
    
    This approach captures semantic uncertainty beyond surface-level variations,
    making it effective for detecting when a model is uncertain about the
    core meaning of its response rather than just lexical choices.
    
    Attributes:
        dependencies (List[str]): Requires 'semantic_classes_entail' statistics
        level (str): Operates at 'sequence' level
        
    Examples:
        >>> from lm_polygraph import WhiteboxModel
        >>> from lm_polygraph.estimators import LabelProb
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> estimator = LabelProb()
        >>> # Note: Requires sampling-based generation with multiple samples
        >>> from lm_polygraph.utils.manager import estimate_uncertainty_batched
        >>> uncertainty = estimate_uncertainty_batched(
        ...     model, estimator, ["Explain quantum physics"],
        ...     generation_params={"do_sample": True, "num_return_sequences": 10}
        ... )
        
    See Also:
        SemanticEntropy: Alternative semantic uncertainty measure
        NumSemSets: Counts number of distinct semantic clusters
    """
    
    def __init__(self):
        """
        Initialize LabelProb estimator with semantic clustering dependency.
        
        The estimator requires semantic clustering to be performed on
        generated samples, which groups semantically similar outputs together.
        """
        super().__init__(["semantic_classes_entail"], "sequence")

    def __str__(self) -> str:
        """Return the unique string identifier for this estimator."""
        return "LabelProb"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty based on semantic cluster distribution.
        
        Uncertainty is computed as the complement of the proportion of samples
        in the largest semantic cluster. This captures how spread out the
        model's outputs are across different semantic meanings.
        
        Parameters:
            stats: Dictionary containing 'semantic_classes_entail' with:
                - 'class_to_sample': Dict mapping class indices to lists of
                  sample indices belonging to that semantic class
                - 'sample_to_class': Dict mapping sample indices to their
                  assigned semantic class
                   
        Returns:
            np.ndarray: Array of uncertainty scores in [0, 1] range where:
                - 0: All samples belong to one semantic class (low uncertainty)
                - Values close to 1: Uniform distribution across many classes
                  (high uncertainty)
                Shape: (n_sequences,)
            
        Raises:
            KeyError: If required semantic clustering data is not in stats
            
        Note:
            This method requires multiple samples to be generated for each
            input to properly estimate the semantic distribution.
        """
        batch_class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        batch_sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]

        ues = []
        for batch_i, class_to_sample in batch_class_to_sample.items():
            num_samples = len(batch_sample_to_class[batch_i])
            largest_class_size = max([len(samples) for samples in class_to_sample])
            # Uncertainty is the complement of the largest class proportion
            ues.append(1 - largest_class_size / num_samples)

        return np.array(ues)
