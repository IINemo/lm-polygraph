import numpy as np
from typing import Dict

from .estimator import Estimator

class LastTokenRepresentationAnalysis(Estimator):
    """
    Analyzes the representation of the last token from the final layer 
    for uncertainty estimation.
    Works only with CausalLM models.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(["layer_wise_pooling"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "LastTokenRepresentationAnalysis"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates uncertainty using the final layer's representation 
        of the last token in the sequence.

        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                * layer_wise_pooling: Dictionary with last token representations from each layer
                
        Returns:
            np.ndarray: Float uncertainty score for each sample.
        """
        layer_embeddings = stats["layer_wise_pooling"]
        
        # Get the final layer's representation
        last_layer_idx = max(int(k.split('_')[1]) for k in layer_embeddings.keys())
        final_layer_last_token = layer_embeddings[f"layer_{last_layer_idx}"]
        
        # Compute uncertainty score from the final layer's last token representation
        uncertainty_scores = np.linalg.norm(final_layer_last_token, axis=1)
        
        return uncertainty_scores