import numpy as np
import torch
from tqdm import tqdm

from typing import Dict

from .estimator import Estimator

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-15, 0, 1)]

def compute_inv_covariance(centroids, train_features, train_labels, jitters=None):
    if jitters is None:
        jitters = JITTERS
    jitter = 0
    jitter_eps = None

    cov = torch.zeros(
        centroids.shape[1], centroids.shape[1], device=centroids.device
    ).float()
    for c, mu_c in tqdm(enumerate(centroids)):
        for x in train_features[train_labels == c]:
            d = (x - mu_c).unsqueeze(1)
            cov += d @ d.T
    cov_scaled = cov / (train_features.shape[0] - 1)

    for i, jitter_eps in enumerate(jitters):
        jitter = jitter_eps * torch.eye(
            cov_scaled.shape[1],
            device=cov_scaled.device,
        )
        cov_scaled_update = cov_scaled + jitter
        eigenvalues = torch.symeig(cov_scaled_update).eigenvalues
        if (eigenvalues >= 0).all():
            break
    cov_scaled = cov_scaled + jitter
    cov_inv = torch.inverse(cov_scaled.to(torch.float64)).float()
    return cov_inv, jitter_eps

def mahalanobis_distance_with_known_centroids_sigma_inv(
    centroids, centroids_mask, sigma_inv, eval_features
):
    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(
        0
    )  # bs (b), num_labels (c / s), dim (d / a)
    dists = torch.sqrt(torch.einsum("bcd,da,bsa->bcs", diff, sigma_inv, diff))
    device = dists.device
    dists = torch.stack([torch.diag(dist).cpu() for dist in dists], dim=0)
    if centroids_mask is not None:
        dists = dists.masked_fill_(centroids_mask, float("inf")).to(device)
    return dists  # np.min(dists, axis=1)

class MahalanobisDistanceSeq(Estimator):
    def __init__(self, embeddings_type: str = "decoder"):
        super().__init__(['embeddings', 'train_embeddings'], 'sequence')
        self.centroid = None
        self.sigma_inv = None
        self.embeddings_type = embeddings_type

    def __str__(self):
        return f'MahalanobisDistanceSeq_{self.embeddings_type}'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        embeddings = stats[f'embeddings_{self.embeddings_type}']       
        if self.centroid is None:
            self.centroid = stats[f'train_embeddings_{self.embeddings_type}'].mean(dim=0)
        if self.sigma_inv is None:
            train_labels = np.zeros(stats[f'train_embeddings_{self.embeddings_type}'].shape[0])
            self.sigma_inv, _ = compute_inv_covariance(
                self.centroid.unsqueeze(0), stats[f'train_embeddings_{self.embeddings_type}'], train_labels
            )
        dists = mahalanobis_distance_with_known_centroids_sigma_inv(
            self.centroid.unsqueeze(0),
            None,
            self.sigma_inv,
            embeddings,
        )[:, 0]
        return dists.cpu().detach().numpy()

# class MutualInformationToken(Estimator):
#     def __init__(self):
#         super().__init__(['embeddings'], 'token')

#     def __str__(self):
#         return 'MutualInformationToken'

#     def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
#         logprobs = stats['greedy_log_likelihoods']
#         lm_logprobs = stats['greedy_lm_log_likelihoods']
#         mi_scores = []
#         for lp, lm_lp in zip(logprobs, lm_logprobs):
#             mi_scores.append([])
#             for t in range(len(lp)):
#                 mi_scores[-1].append(lp[t] - (lm_lp[t - 1] if t > 0 else 0))
#         return np.concatenate([-np.array(sc[:-1]) for sc in mi_scores])
