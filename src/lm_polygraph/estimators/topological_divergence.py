import os
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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

from .estimator import Estimator
from .mtopdiv import (
    transform_attention_scores_to_distances,
    transform_distances_to_mtopdiv,
)


def get_mtopdivs(
    heads: List[Tuple[int, int]],
    length_responses: List[int],
    attention_weights_batch: np.array,
    n_jobs: Optional[float] = -1,
) -> np.ndarray:
    batch_size = attention_weights_batch.shape[0]
    padding_lengths = np.isnan(attention_weights_batch[:, 0, 0, 0]).sum(axis=-1)
    attention_weights_batch = torch.from_numpy(attention_weights_batch)
    distance_matrices_batch = transform_attention_scores_to_distances(
        attention_weights_batch
    ).numpy()

    def job(layer, head):
        mtopdivs = []
        for sample_id in range(batch_size):
            distance_matrice = distance_matrices_batch[sample_id, layer, head]
            padding_length = padding_lengths[sample_id]
            response_length = length_responses[sample_id]
            if padding_length > 0:
                distance_matrice = distance_matrice[:-padding_length, :-padding_length]
            distance_matrice[:-response_length, :-response_length] = 0
            mtopdiv = transform_distances_to_mtopdiv(distance_matrice)
            mtopdivs.append(mtopdiv)
        return np.array(mtopdivs, dtype=float)

    if IS_PARALLEL_AVAILABLE:
        mtopdivs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(job)(layer, head) for layer, head in heads
        )
    else:
        mtopdivs = [job(layer, head) for layer, head in heads]
    mtopdivs = np.stack(mtopdivs, axis=1)
    return mtopdivs


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
                "train_labels",
                "train_mtopdivs",
                "greedy_tokens",
                "forwardpass_attention_weights",
            ]
        else:
            calculators = ["greedy_tokens", "forwardpass_attention_weights"]

        super().__init__(calculators, "sequence")
        self._selected_heads = heads
        self._best_heads = None
        self._max_heads = max_heads
        self._n_jobs = n_jobs

    def __str__(self):
        return "TopologicalDivergence"

    @property
    def best_heads(self):
        return self._best_heads

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

        if self._selected_heads is None and self._best_heads is None:
            train_mtopdivs = stats["train_mtopdivs"]
            train_labels = stats["train_labels"]
            best_heads = self._select_heads(
                train_mtopdivs,
                train_labels,
            )
            _, num_layers, num_heads, _, _ = attention_weights_batch.shape
            best_heads = np.unravel_index(best_heads, (num_layers, num_heads))
            best_heads = list(zip(best_heads[0], best_heads[1]))
            self._best_heads = best_heads

            # After selecting heads once, we drop train stats to prevent recomputing them in later runs.
            # This assumes the manager object is reinitialized before the next estimator call.
            # Alternatively, a new estimator instance can be created with the selected heads.
            self.stats_dependencies = ["greedy_tokens", "forwardpass_attention_weights"]

        mtopdivs = get_mtopdivs(
            self._best_heads if self._best_heads else self._selected_heads,
            length_responses,
            attention_weights_batch,
            n_jobs=self._n_jobs,
        )

        return np.mean(mtopdivs, axis=1)

    def _select_heads(self, scores, labels):
        grounded_scores, hal_scores = scores[labels == 0], scores[labels == 1]
        deltas = hal_scores.mean(0) - grounded_scores.mean(0)
        heads = sorted(range(len(deltas)), key=lambda x: deltas[x], reverse=True)

        best_auroc, n_opt = 0, 0
        for n in range(1, self._max_heads + 1):
            n_best_heads = heads[:n]
            predictions = scores[:, n_best_heads].mean(axis=1)
            roc_auc = roc_auc_score(labels, predictions)
            if roc_auc > best_auroc:
                best_auroc = roc_auc
                n_opt = n
        return heads[:n_opt]
