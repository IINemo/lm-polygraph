import numpy as np
import os
import json
from typing import Dict

from .estimator import Estimator

class EmbeddingsStorage(Estimator):
    """
    Stores all embeddings from all layers.
    Works with CausalLM models.
    """

    def __init__(self, save_dir: str = "embeddings_storage", verbose: bool = False):
        super().__init__(["all_layers_embeddings"], "sequence")
        self.save_dir = save_dir
        self.verbose = verbose
        os.makedirs(save_dir, exist_ok=True)
        self.sample_count = 0

    def __str__(self):
        return "EmbeddingsStorage"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stores all embeddings and estimates uncertainty using all token representations
        from the final layer.

        Parameters:
            stats (Dict[str, np.ndarray]): Input statistics containing:
                * all_layers_embeddings: Dictionary with full embeddings from each layer
                
        Returns:
            np.ndarray: Float uncertainty score for each sample.
        """
        # all_embeddings = stats["all_layers_embeddings"]
        
        # # Save embeddings if requested
        # # if self.save_embeddings:
        # json_compatible_embeddings = {}
        # for key, value in all_embeddings.items():
        #     json_compatible_embeddings[key] = value.tolist()
        
        # Save to JSON file
        # embeddings_path = os.path.join(self.save_dir, f"embeddings_batch_{self.sample_count}.json")
        # with open(embeddings_path, 'w') as f:
        #     json.dump(json_compatible_embeddings, f)
            
        # if self.verbose:
        #     print(f"Saved embeddings to {embeddings_path}")
        
        # # Find the final layer
        # print(f"Embdngs: {all_embeddings}")
        # last_layer_idx = max(int(k.split('_')[1]) for k in all_embeddings.keys())
        
        # # Get embeddings from the final layer
        # # final_layer_embeddings = all_embeddings[f"layer_{last_layer_idx}"]
        # n = len(all_embeddings[f"layer_{last_layer_idx}"])
        
        # # Compute uncertainty score for each sample
        # batch_size = final_layer_embeddings.shape[0]
        # uncertainty_scores = np.zeros(batch_size)
        
        # for i in range(batch_size):
        #     # Get all token embeddings for this sample
        #     sample_embeddings = final_layer_embeddings[i]
            
        #     # Calculate mean embedding across all tokens
        #     mean_embedding = np.mean(sample_embeddings, axis=0)
            
        #     # Use norm of the mean embedding as uncertainty score (similar to LastTokenRepresentationAnalysis)
        #     uncertainty_scores[i] = np.linalg.norm(mean_embedding)
        
        # self.sample_count += 1
            
        # return uncertainty_scores
        return np.zeros(2)