import os
import warnings
from typing import Dict, List, Tuple
import numpy as np
import torch


try:
    from joblib import Parallel, delayed
    IS_PARALLEL_AVAILABLE = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except ImportError:
    IS_PARALLEL_AVAILABLE = False
    warnings.warn(
        "Joblib is not installed. Parallel processing for TopologicalDivergence call will not be available. "
        "Please install it via 'pip install joblib' if you want to use parallel processing."
        )
try:
    from ripser import ripser
except ImportError:
    raise ImportError(
        "Please install the 'ripser' package to use TopologicalDivergence estimator. "
        "You can install it via 'pip install ripser'."
    )

from .estimator import Estimator


def transform_attention_scores_to_distances(
    attention_weights: np.array
) -> np.array:
    """Transform attention matrix to the matrix of distances between tokens.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Attention matrixes of one sample (n_heads x n_tokens x n_tokens).

    Returns
    -------
    np.array
        Distance matrix.

    """
    attention_weights = torch.from_numpy(attention_weights).float()
    n_tokens = attention_weights.shape[-1]
    distance_mx = 1 - torch.clamp(
        attention_weights, min=0.0
    )
    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    distance_mx *= zero_diag.to(attention_weights.device).expand_as(
        distance_mx
    )
    distance_mx = torch.minimum(distance_mx.transpose(-1, -2), distance_mx)
    return distance_mx.cpu().numpy()


def transform_distances_to_mtopdiv(distance_mx: np.ndarray) -> float:
    """
    Compute the MTopDiv (Manifold Topology Divergence) score from a distance matrix.

    This function calculates the sum of persistence intervals in the H₀ (zero-dimensional)
    persistent homology barcode, corresponding to the lifetimes of connected components
    in a Vietoris–Rips filtration.

    Parameters:
        distance_mx (np.ndarray): A square, symmetric distance matrix.

    Returns:
        float: Sum of finite H₀ barcode lengths (birth–death), representing topological diversity.
    """
    barcodes = ripser(distance_mx, distance_matrix=True, maxdim=0)['dgms']
    if len(barcodes) > 0:
        return barcodes[0][:-1, 1].sum()
    return 0


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
            selected_heads: List[Tuple[int, int]] = None,
            n_jobs: int = 16,
    ):
        """
        Initializes TopologicalDivergence estimator.

        Parameters:
            selected_heads (List[Tuple[int, int]]): List of attention heads to calculate MTopDiv for.
                First integer is layer index, second is head index.
                If not provided or empty, all heads will be used.
            n_jobs (int): Number of jobs for parallel processing. Default: 16.
        """
        super().__init__(["greedy_tokens", "forwardpass_attention_weights"], "sequence")

        self.selected_heads = selected_heads
        self.n_jobs = n_jobs

    def __str__(self):
        return "TopologicalDivergence"

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
        responses = stats["greedy_tokens"]
        attention_weights_batch = stats["forwardpass_attention_weights"]
        batch_size = attention_weights_batch.shape[0]

        if not self.selected_heads:
            num_layers = attention_weights_batch.shape[1]
            num_heads = attention_weights_batch.shape[2]
            self.selected_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]

        padding_lengths = np.isnan(attention_weights_batch[:, 0, 0, 0]).sum(axis=-1)

        distance_matrices_batch = transform_attention_scores_to_distances(
            attention_weights_batch
        )

        def compute_mtopdiv(layer, head):
            mtopdivs = []
            for sample_id in range(batch_size):
                distance_matrice = distance_matrices_batch[sample_id, layer, head]
                padding_length = padding_lengths[sample_id]
                response_length = len(responses[sample_id])
                if padding_length > 0:
                    distance_matrice = distance_matrice[:-padding_length, :-padding_length]
                distance_matrice[:-response_length, :-response_length] = 0
                mtopdiv = transform_distances_to_mtopdiv(distance_matrice)
                mtopdivs.append(mtopdiv)
            return np.array(mtopdivs, dtype=float)

        if IS_PARALLEL_AVAILABLE:
            mtopdivs = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(compute_mtopdiv)(layer, head)
                for layer, head in self.selected_heads
            )
        else:
            mtopdivs = [
                compute_mtopdiv(layer, head)
                for layer, head in self.selected_heads
            ]
        mtopdivs = np.stack(mtopdivs, axis=1)
        return np.mean(mtopdivs, axis=1)
