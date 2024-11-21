import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch
from torch.nn import functional as F

class EntropyCalculator(StatCalculator):
    """
    Calculates entropy of probabilities at each token position in the generation of a Whitebox model.
    """

    def __init__(self):
        super().__init__(["entropy"], ["greedy_log_probs"])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str] = None,
        model: WhiteboxModel = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the entropy of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, which includes:
                * 'greedy_log_probs' (List[List[float]]): log-probabilities of the generation tokens.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] entropies calculated at 'entropy' key.
        """
        logprobs = dependencies["greedy_log_probs"]
        entropies = []
        for s_lp in logprobs:
            entropies.append([])
            for lp in s_lp:
                mask = ~np.isinf(lp)
                entropies[-1].append(-np.sum(np.array(lp[mask]) * np.exp(lp[mask])))
        return {"entropy": entropies}

class SampleEntropyCalculator(StatCalculator):
    def __init__(self):
        super().__init__(["sample_entropy"], ["token_distributions"])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str] = None,
        model: WhiteboxModel = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        batch_distributions = dependencies["token_distributions"]
        entropies = []
        
        for input_distributions in batch_distributions:
            for sample_distributions in input_distributions:
                sample_entropies = []
                for token_dist in sample_distributions:
                    # Convert token_dist to a numpy array first, then to a torch tensor
                    token_dist_tensor = torch.tensor(np.array(token_dist))

                    # Calculate entropy using torch's Categorical distribution
                    entropy = torch.distributions.Categorical(probs=token_dist_tensor).entropy()
                    sample_entropies.append(entropy.item()) 

            # Calculate mean entropy for the sample
            mean_entropy = torch.mean(torch.tensor(sample_entropies)) if sample_entropies else 0
            entropies.append(mean_entropy.item())
        
        return {"sample_entropy": entropies}
