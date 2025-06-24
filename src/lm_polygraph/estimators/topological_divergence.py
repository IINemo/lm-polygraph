from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

from .estimator import Estimator
from .mtopdiv import get_mtopdivs


class TopologicalDivergence(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Hallucination Detection in LLMs with Topological Divergence on Attention Graphs"
    as provided in the paper https://arxiv.org/abs/2504.10063.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    Computes topological divergences between prompt and response attention
    graphs to identify hallucination-indicative heads.
    """

    def __init__(
        self,
        heads: Optional[List[Tuple[int, int]]] = None,
        max_heads: Optional[int] = 6,
        n_jobs: int = -1,
    ):
        """
        Initializes TopologicalDivergence estimator.

        Parameters:
            selected_heads (List[Tuple[int, int]]): List of attention heads to calculate MTopDiv for.
                First integer is layer index, second is head index.
                If not provided or empty, all heads will be used.
            n_jobs (int): Number of jobs for parallel processing. Default: 16.
        """
        if not heads:
            calculators = [
                "topological_divergence_heads",
                "greedy_tokens",
                "forwardpass_attention_weights",
            ]
        else:
            calculators = ["greedy_tokens", "forwardpass_attention_weights"]

        super().__init__(calculators, "sequence")
        self._heads = heads
        self._max_heads = max_heads
        self._n_jobs = n_jobs

    def __str__(self):
        return "TopologicalDivergence"

    @property
    def heads(self):
        return self._heads

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates sequence-wise MTopDiv scores for selected heads of attention masks.
        Returns the mean MTopDiv score for each input text.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, consisting of:
                * tokenized model generations for each input text in 'greedy_tokens'
                * attention scores of shape [batch_size, num_layers, num_heads, seq_len, seq_len]
                    in 'forwardpass_attention_weights'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        length_responses = list(map(len, stats["greedy_tokens"]))
        attention_weights_batch = stats["forwardpass_attention_weights"]

        if self._heads is None:
            best_heads = stats["topological_divergence_heads"]
            self._heads = best_heads

            # After selecting heads once, we drop train stats to prevent recomputing them in later runs.
            # This assumes the manager object is reinitialized before the next estimator call.
            # Alternatively, a new estimator instance can be created with the selected heads.
            self.stats_dependencies = ["greedy_tokens", "forwardpass_attention_weights"]

        attention_weights_batch = torch.from_numpy(attention_weights_batch)
        mtopdivs = get_mtopdivs(
            self._heads,
            length_responses,
            attention_weights_batch,
            n_jobs=self._n_jobs,
        )

        return np.mean(mtopdivs, axis=1)


