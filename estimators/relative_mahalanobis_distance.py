import os
import numpy as np
import torch
from tqdm import tqdm

from typing import Dict

from .estimator import Estimator
from .mahalanobis_distance import compute_inv_covariance, mahalanobis_distance_with_known_centroids_sigma_inv, MahalanobisDistanceSeq

def save_array(array, filename):
    with open(filename, 'wb') as f:
        np.save(f, array)
        
def load_array(filename):
    with open(filename, 'rb') as f:
        array = np.load(f)
    return array


class RelativeMahalanobisDistanceSeq(Estimator):
    def __init__(self, embeddings_type: str = "decoder", parameters_path: str = None, normalize: bool = False):
        super().__init__(['embeddings', 'train_embeddings', 'background_train_embeddings'], 'sequence')
        self.centroid_0 = None
        self.sigma_inv_0 = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e+100
        self.max = -1e+100
        self.MD = MahalanobisDistanceSeq(embeddings_type, parameters_path, normalize=False)
        
        if (self.parameters_path is not None) and os.path.exists(f"{self.parameters_path}/centroid_0.pt"):
            self.centroid_0 = torch.load(f"{self.parameters_path}/centroid_0.pt")
            self.sigma_inv_0 = torch.load(f"{self.parameters_path}/sigma_inv_0.pt")
            self.max = load_array(f"{self.parameters_path}/max_0.npy")
            self.min = load_array(f"{self.parameters_path}/min_0.npy")

    def __str__(self):
        return f'RelativeMahalanobisDistanceSeq_{self.embeddings_type}'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        embeddings = stats[f'embeddings_{self.embeddings_type}']   
        
        if self.centroid_0 is None:
            self.centroid_0 = stats[f'background_train_embeddings_{self.embeddings_type}'].mean(dim=0)
            if self.parameters_path is not None:
                if not os.path.exists(f"{self.parameters_path}"):
                    os.mkdir(self.parameters_path)
                torch.save(self.centroid_0, f"{self.parameters_path}/centroid_0.pt")
                
        if self.sigma_inv_0 is None:
            train_labels = np.zeros(stats[f'background_train_embeddings_{self.embeddings_type}'].shape[0])
            self.sigma_inv_0, _ = compute_inv_covariance(
                self.centroid_0.unsqueeze(0), stats[f'background_train_embeddings_{self.embeddings_type}'], train_labels
            )
            if self.parameters_path is not None:
                torch.save(self.sigma_inv_0, f"{self.parameters_path}/sigma_inv_0.pt")
                
        dists_0 = mahalanobis_distance_with_known_centroids_sigma_inv(
            self.centroid_0.unsqueeze(0),
            None,
            self.sigma_inv_0,
            embeddings,
        )[:, 0].cpu().detach().numpy()
        
        md = self.MD(stats)
        
        dists = md - dists_0
        if self.max < dists.max():
            self.max = dists.max()
            if self.parameters_path is not None:
                save_array(self.max, f"{self.parameters_path}/max_0.npy")
        if self.min > dists.min():
            self.min = dists.min()
            if self.parameters_path is not None:
                save_array(self.min, f"{self.parameters_path}/min_0.npy")
                
        if self.normalize:
            dists = np.clip((self.max - dists) / (self.max - self.min), a_min=0, a_max=1) 
                
        return dists