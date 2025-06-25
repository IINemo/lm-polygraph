import numpy as np
from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    create_cuda_tensor_from_numpy,
)


class TokenMahalanobisDistance(Estimator):
    """
    Token Mahalanobis Distance (TMD) as described in https://aclanthology.org/2025.naacl-long.113/.

    This method computes the Token Mahalanobis distance between each token embedding and the centroid
    of in-distribution token embeddings, using the inverse covariance estimated from the high-quality training data.
    The resulting token-level distances can be aggregated (e.g., mean) to produce a sequence-level
    uncertainty score.

    Args:
        embeddings_type (str): Which embeddings to use ("decoder" or "encoder").
        layer (int): Which hidden layer to use (-1 for last).
        metric_thr (float): Threshold for binarizing the metric (used for filtering high-quality training data).
        aggregation (str): How to aggregate token-level scores into a sequence-level score ("mean" or "none").
        device (str): Device for computation ("cuda" or "cpu").
        storage_device (str): Device for storing centroids/covariances ("cuda" or "cpu").
    """
    def __init__(
        self,
        embeddings_type: str = "decoder",
        layer: int = -1,
        metric_thr: float = 0.3,
        aggregation: str = "mean",
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        super().__init__(
            [
                "token_embeddings",
                "train_token_embeddings",
                "train_metrics",
            ],
            "sequence",
        )
        self.layer = layer
        self.layer_name = "" if self.layer == -1 else f"_{self.layer}"
        self.embeddings_type = embeddings_type
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.device = device
        self.storage_device = storage_device
        self.aggregation = aggregation

    def __str__(self):
        return f"TokenMahalanobisDistance_{self.embeddings_type}{self.layer_name}"

    def __call__(
        self, stats: Dict[str, np.ndarray], save_data: bool = True
    ) -> np.ndarray:

        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}{self.layer_name}"]
        )
        if not self.is_fitted:
            train_metrics = stats["train_metrics"]
            train_greedy_tokens = stats["train_greedy_tokens"]
            token_level_metrics = np.concatenate(
                [
                    np.full(len(tokens), metric)
                    for metric, tokens in zip(train_metrics, train_greedy_tokens)
                ]
            )

            centroid_key = (
                f"centroid{self.layer_name}_{self.metric_thr}_{len(train_metrics)}"
            )
            if (
                centroid_key in stats.keys()
            ):  # to reduce number of stored centroid for multiple methods used the same data
                self.centroid = stats[centroid_key]
                if self.storage_device == "cpu":
                    self.centroid = self.centroid.cpu()
                elif self.storage_device == "cuda":
                    self.centroid = self.centroid.cuda()
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[
                        f"train_token_embeddings_{self.embeddings_type}{self.layer_name}"
                    ]
                )
                if self.metric_thr > 0:
                    # filter out tokens with metrics below threshold only if there are more than 10 tokens
                    if (token_level_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[
                            token_level_metrics >= self.metric_thr
                        ]
                self.centroid = train_embeddings.mean(axis=0)
                if self.storage_device == "cpu":
                    self.centroid = self.centroid.cpu()

                if save_data:
                    stats[centroid_key] = self.centroid

            covariance_key = (
                f"covariance{self.layer_name}_{self.metric_thr}_{len(train_metrics)}"
            )
            if covariance_key in stats.keys():
                self.sigma_inv = stats[covariance_key]
                if self.storage_device == "cpu":
                    self.sigma_inv = self.sigma_inv.cpu()
                elif self.storage_device == "cuda":
                    self.sigma_inv = self.sigma_inv.cuda()
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[
                        f"train_token_embeddings_{self.embeddings_type}{self.layer_name}"
                    ]
                )
                if self.metric_thr > 0:
                    if (token_level_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[
                            token_level_metrics >= self.metric_thr
                        ]
                self.sigma_inv, _ = compute_inv_covariance(
                    self.centroid.unsqueeze(0), train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv = self.sigma_inv.cpu()

                if save_data:
                    stats[covariance_key] = self.sigma_inv
            self.is_fitted = True

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.float(),
                    None,
                    self.sigma_inv.float(),
                    embeddings.cpu().float(),
                )[:, 0]
            else:
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.cuda().float(),
                    None,
                    self.sigma_inv.cuda().float(),
                    embeddings.float(),
                )[:, 0]
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                self.centroid.float(),
                None,
                self.sigma_inv.float(),
                embeddings.float(),
            )[:, 0]
        else:
            raise NotImplementedError

        k = 0
        agg_dists = []
        greedy_tokens = stats["greedy_tokens"]
        for tokens in greedy_tokens:
            dists_i = dists[k : k + len(tokens)].cpu().detach().numpy()
            k += len(tokens)

            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))

        if self.aggregation == "none":
            agg_dists = dists.cpu().detach().numpy()
        else:
            agg_dists = np.array(agg_dists)

        return agg_dists
