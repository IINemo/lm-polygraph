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

    def __init__(
        self,
        top_k: int = None,
    ):
        self.top_k = top_k
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
                lp = torch.tensor(lp)
                if self.top_k is not None:
                    lp = torch.topk(lp, self.top_k).values
                #mask = ~np.isinf(lp)
                #lp = lp[mask]
                #if self.top_k is not None:
                #    lp = np.sort(lp)[-self.top_k:]
                #entropies[-1].append(-np.sum(np.array(lp) * np.exp(lp)))
                entropies[-1].append(torch.distributions.Categorical(logits=lp).entropy().item())
        return {"entropy": entropies}

class SampleEntropyCalculator(StatCalculator):
    def __init__(
        self,
        top_k: int = None,
    ):
        self.top_k = top_k
        super().__init__(["sample_entropy"], ["sample_tokens_distributions"])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str] = None,
        model: WhiteboxModel = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        batch_distributions = dependencies["sample_tokens_distributions"]

        input_entropies = []
        for input_distributions in batch_distributions:
            sample_entropies = []
            for sample_distributions in input_distributions:
                token_entropies = []
                for token_dist in sample_distributions:
                    # Convert token_dist to a numpy array first, then to a torch tensor
                    token_dist_tensor = torch.tensor(token_dist)

                    if self.top_k is not None:
                        token_dist_tensor = torch.topk(token_dist_tensor, self.top_k).values

                    # Calculate entropy using torch's Categorical distribution
                    entropy = torch.distributions.Categorical(logits=token_dist_tensor).entropy()
                    token_entropies.append(entropy.item()) 

                # Calculate mean entropy for the sample
                sample_entropy = torch.mean(torch.tensor(token_entropies)) if token_entropies else 0
                sample_entropies.append(sample_entropy.item())
            input_entropies.append(sample_entropies)

        return {"sample_entropy": input_entropies}
