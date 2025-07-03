import os
import yaml
import warnings
import numpy as np
from typing import List, Tuple, Optional
import ast

try:
    from ripser import ripser
except ImportError:
    raise ImportError(
        "Please install the 'ripser' package to use TopologicalDivergence estimator. "
        "You can install it via 'pip install ripser'."
    )

try:
    from joblib import Parallel, delayed

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    IS_PARALLEL_AVAILABLE = True
except ImportError:
    IS_PARALLEL_AVAILABLE = False
    warnings.warn(
        "Joblib is not installed. Parallel processing for TopologicalDivergence call will not be available. "
        "Please install it via 'pip install joblib' if you want to use parallel processing."
    )


def transform_attention_scores_to_distances(
    attention_weights: np.ndarray,
) -> np.ndarray:
    """Transform attention matrix to the matrix of distances between tokens.

    Parameters
    ----------
    attention_weights : np.ndarray
        Attention matrices of one sample (n_heads x n_tokens x n_tokens).

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    attention_weights = attention_weights.astype(np.float32)
    n_tokens = attention_weights.shape[-1]
    distance_mx = 1 - np.clip(attention_weights, a_min=0.0, a_max=None)
    zero_diag = np.ones((n_tokens, n_tokens)) - np.eye(n_tokens)

    distance_mx *= np.broadcast_to(zero_diag, distance_mx.shape)
    distance_mx = np.minimum(
        np.swapaxes(distance_mx, -1, -2),
        distance_mx,
    )

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


def get_mtopdivs(
    heads: List[Tuple[int, int]],
    length_responses: List[int],
    attention_weights_batch: np.array,
    n_jobs: Optional[float] = -1,
) -> np.ndarray:
    batch_size = attention_weights_batch.shape[0]
    padding_lengths = np.isnan(attention_weights_batch[:, 0, 0, 0]).sum(axis=-1)

    def job(layer_head_pair):
        layer, head = layer_head_pair
        mtopdivs = []
        distance_matrices = transform_attention_scores_to_distances(
            attention_weights_batch[:, layer, head]
        )

        for sample_id in range(batch_size):
            distance_matrix = distance_matrices[sample_id]
            padding_length = padding_lengths[sample_id]
            response_length = length_responses[sample_id]

            if padding_length > 0:
                distance_matrix = distance_matrix[:-padding_length, :-padding_length]
            distance_matrix[:-response_length, :-response_length] = 0
            mtopdiv = transform_distances_to_mtopdiv(distance_matrix) / response_length
            mtopdivs.append(mtopdiv)

        return np.array(mtopdivs, dtype=float)

    if IS_PARALLEL_AVAILABLE:
        with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
            mtopdivs = parallel(delayed(job)((layer, head)) for layer, head in heads)
    else:
        mtopdivs = [job((layer, head)) for layer, head in heads]

    mtopdivs = np.stack(mtopdivs, axis=1)
    return mtopdivs


def load_model_heads(
    cache_path: Optional[str],
    model_path: str,
) -> Optional[List[Tuple[int, int]]]:
    """
    Load model heads.

    Parameters
    ----------
    cache_path : Optional[str]
        Path to the YAML cache file.
    model_path : str
        Unique path to the model.

    Returns
    -------
    Optional[List[Tuple[int, int]]]
        List of heads for the model if found, else None.
    """
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                config = yaml.safe_load(f)
            models = config.get("models", [])
            for model_entry in models:
                if isinstance(model_entry, dict) and model_path in model_entry:
                    return ast.literal_eval(model_entry[model_path])

            print(f"Model '{model_path}' not found in cache.")
        except Exception as e:
            print(f"Failed to load heads from cache: {e}")
    return None


def save_model_heads(
    cache_path: str,
    model_path: str,
    heads: List[Tuple[int, int]],
) -> None:
    """
    Save a list of heads for a model.

    Parameters
    ----------
    cache_path : str
        Path to the YAML file to save the model heads.
    model_path : str
        Unique path to the model.
    heads : List[Tuple[int, int]]
        List of heads to save.
    """
    if os.path.isfile(cache_path):
        with open(cache_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    models = config.get("models", [])

    for entry in models:
        if isinstance(entry, dict) and model_path in entry:
            return

    models.append({model_path: f"{heads}"})
    config["models"] = models

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    with open(cache_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Saved heads for '{model_path}' to {cache_path}")
