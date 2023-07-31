import os
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
        eigenvalues = torch.symeig(cov_scaled_update.cpu()).eigenvalues
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
    def __init__(self, embeddings_type: str = "decoder", parameters_path: str = None, normalize: bool = False):
        super().__init__(['embeddings', 'train_embeddings'], 'sequence')
        self.centroid = None
        self.sigma_inv = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e+100
        self.max = -1e+100
        
        if (self.parameters_path is not None) and os.path.exists(f"{self.parameters_path}/centroid.pt"):
            self.centroid = torch.load(f"{self.parameters_path}/centroid.pt")
            self.sigma_inv = torch.load(f"{self.parameters_path}/sigma_inv.pt")
            self.max = torch.load(f"{self.parameters_path}/max.pt")
            self.min = torch.load(f"{self.parameters_path}/min.pt")
            

    def __str__(self):
        return f'MahalanobisDistanceSeq_{self.embeddings_type}'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        embeddings = stats[f'embeddings_{self.embeddings_type}']       
        if self.centroid is None:
            self.centroid = stats[f'train_embeddings_{self.embeddings_type}'].mean(dim=0)
            if self.parameters_path is not None:
                if not os.path.exists(f"{self.parameters_path}"):
                    os.mkdir(self.parameters_path)
                torch.save(self.centroid, f"{self.parameters_path}/centroid.pt")
                
        if self.sigma_inv is None:
            train_labels = np.zeros(stats[f'train_embeddings_{self.embeddings_type}'].shape[0])
            self.sigma_inv, _ = compute_inv_covariance(
                self.centroid.unsqueeze(0), stats[f'train_embeddings_{self.embeddings_type}'], train_labels
            )
            if self.parameters_path is not None:
                torch.save(self.sigma_inv, f"{self.parameters_path}/sigma_inv.pt")
                
        dists = mahalanobis_distance_with_known_centroids_sigma_inv(
            self.centroid.unsqueeze(0),
            None,
            self.sigma_inv,
            embeddings,
        )[:, 0]
        
        if self.max < dists.max():
            self.max = dists.max()
            if self.parameters_path is not None:
                torch.save(self.max, f"{self.parameters_path}/max.pt")
        if self.min > dists.min():
            self.min = dists.min()
            if self.parameters_path is not None:
                torch.save(self.min, f"{self.parameters_path}/min.pt")
                
        if self.normalize:
            dists = torch.clip((self.max - dists) / (self.max - self.min), min=0, max=1) 
                
        return dists.cpu().detach().numpy()