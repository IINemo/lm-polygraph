import os
import numpy as np
import torch
from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    create_cuda_tensor_from_numpy,
)
from .token_mahalanobis_distance import TokenMahalanobisDistance


class RelativeTokenMahalanobisDistance(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        layer: int = -1,
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        self.layer = layer
        self.layer_name = "" if self.layer == -1 else f"_{self.layer}"
        super().__init__(
            [
                "token_embeddings",
                "train_token_embeddings",
                "background_train_token_embeddings",
                "train_metrics",
            ],
            "sequence",
        )

        self.embeddings_type = embeddings_type
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.device = device
        self.storage_device = storage_device
        self.is_fitted = False

        self.MD = TokenMahalanobisDistance(
            embeddings_type=embeddings_type,
            layer=layer,
            metric_thr=metric_thr,
            aggregation="none",
            device=device,
            storage_device=storage_device,
        )

    def __str__(self):
        return (
            f"RelativeTokenMahalanobisDistance_{self.embeddings_type}{self.layer_name}"
        )

    def __call__(
        self, stats: Dict[str, np.ndarray], save_data: bool = True
    ) -> np.ndarray:
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}{self.layer_name}"]
        )

        if not self.is_fitted:
            train_greedy_texts = stats["train_greedy_texts"]
            centroid_key = f"background_centroid{self.layer_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if centroid_key in stats.keys():
                self.centroid_0 = stats[centroid_key]
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()
                elif self.storage_device == "cuda":
                    self.centroid_0 = self.centroid_0.cuda()
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[
                        f"background_train_token_embeddings_{self.embeddings_type}{self.layer_name}"
                    ]
                )
                self.centroid_0 = background_train_embeddings.mean(axis=0)
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()

                if save_data:
                    stats[centroid_key] = self.centroid_0

            covariance_key = f"background_covariance{self.layer_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if covariance_key in stats.keys():
                self.sigma_inv_0 = stats[covariance_key]
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()
                elif self.storage_device == "cuda":
                    self.sigma_inv_0 = self.sigma_inv_0.cuda()
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[
                        f"background_train_token_embeddings_{self.embeddings_type}{self.layer_name}"
                    ]
                )
                self.sigma_inv_0, _ = compute_inv_covariance(
                    self.centroid_0.unsqueeze(0), background_train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()

                if save_data:
                    stats[covariance_key] = self.sigma_inv_0

            self.is_fitted = True

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.cpu().float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.cuda().float(),
                        None,
                        self.sigma_inv_0.cuda().float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists_0 = (
                mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid_0.float(),
                    None,
                    self.sigma_inv_0.float(),
                    embeddings.float(),
                )[:, 0]
                .cpu()
                .detach()
                .numpy()
            )
        else:
            raise NotImplementedError

        # Compute original Mahalanobis distances
        md = self.MD(stats, save_data=save_data)

        # Relative Mahalanobis Distance (RMD)
        dists = md - dists_0

        agg_dists = []
        k = 0
        greedy_tokens = stats["greedy_tokens"]
        for tokens in greedy_tokens:
            dists_i = dists[k : k + len(tokens)]
            k += len(tokens)
            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))

        if self.aggregation == "none":
            agg_dists = dists

        agg_dists = np.array(agg_dists)

        return agg_dists
