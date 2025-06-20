import numpy as np
import torch

try:
    from ripser import ripser
except ImportError:
    raise ImportError(
        "Please install the 'ripser' package to use TopologicalDivergence estimator. "
        "You can install it via 'pip install ripser'."
    )


def transform_attention_scores_to_distances(
    attention_weights: torch.Tensor,
) -> torch.Tensor:
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
    attention_weights = attention_weights.float()
    n_tokens = attention_weights.shape[-1]
    distance_mx = 1 - torch.clamp(attention_weights, min=0.0)
    zero_diag = torch.ones(n_tokens, n_tokens) - torch.eye(n_tokens)
    zero_diag = zero_diag.to(attention_weights.device)
    distance_mx *= zero_diag.expand_as(distance_mx)
    distance_mx = torch.minimum(distance_mx.transpose(-1, -2), distance_mx)
    return distance_mx


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
    barcodes = ripser(distance_mx, distance_matrix=True, maxdim=0)["dgms"]
    if len(barcodes) > 0:
        return barcodes[0][:-1, 1].sum()
    return 0
