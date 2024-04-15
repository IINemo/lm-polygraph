import os
import numpy as np
import torch

from typing import Dict

from .estimator import Estimator

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-15, 0, 1)]


def compute_inv_covariance(centroids, train_features, jitters=None):
    r"""
    This function computes inverse covariance matrix that is required by Mahalanobis distance:
    MD = \sqrt((h(x) - \mu)^{T} \Sigma^{-1} (h(x) - \mu))

    """

    # jitter is the value to be added to the covariance matrix

    if jitters is None:
        jitters = JITTERS
    jitter = 0
    jitter_eps = None

    # A nested loop iterates over each centroid (mu_c) and the corresponding training features (x) for that centroid.
    # and for each pair of centroid and feature, the difference (d) between the feature and centroid is computed and
    # the outer product of d with itself is added to the covariance matrix.

    if torch.cuda.is_available():
        centroids = centroids.cuda()
        train_features = train_features.cuda()

    cov_scaled = torch.cov(train_features.T)

    # The function then iterates over each jitter_eps value in jitters and adds jitter to the scaled covariance matrix.
    # And the eigenvalues of the updated covariance matrix are computed, and if all eigenvalues are non-negative, the loop breaks.

    for i, jitter_eps in enumerate(jitters):
        jitter = jitter_eps * torch.eye(
            cov_scaled.shape[1],
            device=cov_scaled.device,
        )
        cov_scaled_update = cov_scaled + jitter
        eigenvalues = torch.linalg.eigh(cov_scaled_update).eigenvalues
        if (eigenvalues >= 0).all():
            break
    cov_scaled = cov_scaled + jitter

    # finally computes inverse of scaled covariance matrix with regularisation for MD calculation

    cov_inv = torch.inverse(cov_scaled.to(torch.float64)).float()
    return cov_inv, jitter_eps


def mahalanobis_distance_with_known_centroids_sigma_inv(
    centroids, centroids_mask, sigma_inv, eval_features
):
    """
    - This function takes in centroids, centroids_mask, sigma_inv, and eval_features.
    - tensor of Mahalanobis distances is returned.
    """
    # step 1: calculate the difference (diff) between each evaluation feature and each centroid by subtracting the centoids from the features.

    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(
        0
    )  # bs (b), num_labels (c / s), dim (d / a)

    # step 2: the Mahalanobis distance is computed using the formula: sqrt(diff @ sigmainv @ diff),
    #  where diff is reshaped to match the dimensions of sigmainv.

    dists = torch.sqrt(torch.einsum("bcd,da,bsa->bcs", diff, sigma_inv, diff))
    device = dists.device

    # step 3: obtain a tensor of distances for each evaluation feature and centroid pair.

    dists = torch.stack([torch.diag(dist).cpu() for dist in dists], dim=0)

    # If centroids_mask is not None, the distances corresponding to masked centroids are filled with infinity.

    if centroids_mask is not None:
        dists = dists.masked_fill_(centroids_mask, float("inf")).to(device)
    return dists  # np.min(dists, axis=1)


def create_cuda_tensor_from_numpy(array):
    if not isinstance(array, torch.Tensor):
        array = torch.from_numpy(array)
    if torch.cuda.is_available():
        array = array.cuda()
    return array


class MahalanobisDistanceSeq(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
    ):
        super().__init__(["embeddings", "train_embeddings"], "sequence")
        self.centroid = None
        self.sigma_inv = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.is_fitted = False

        if self.parameters_path is not None:
            self.full_path = f"{self.parameters_path}/md_{self.embeddings_type}"
            os.makedirs(self.full_path, exist_ok=True)

            if os.path.exists(f"{self.full_path}/centroid.pt"):
                self.centroid = torch.load(f"{self.full_path}/centroid.pt")
                self.sigma_inv = torch.load(f"{self.full_path}/sigma_inv.pt")
                self.max = torch.load(f"{self.full_path}/max.pt")
                self.min = torch.load(f"{self.full_path}/min.pt")
                self.is_fitted = True

    def __str__(self):
        return f"MahalanobisDistanceSeq_{self.embeddings_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take the embeddings
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"embeddings_{self.embeddings_type}"]
        )

        # compute centroids if not given
        if not self.is_fitted:
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_embeddings_{self.embeddings_type}"]
            )
            self.centroid = train_embeddings.mean(axis=0)
            if self.parameters_path is not None:
                torch.save(self.centroid, f"{self.full_path}/centroid.pt")

        # compute inverse covariance matrix if not given
        if not self.is_fitted:
            train_embeddings = create_cuda_tensor_from_numpy(
                stats[f"train_embeddings_{self.embeddings_type}"]
            )
            self.sigma_inv, _ = compute_inv_covariance(
                self.centroid.unsqueeze(0), train_embeddings
            )
            if self.parameters_path is not None:
                torch.save(self.sigma_inv, f"{self.full_path}/sigma_inv.pt")
            self.is_fitted = True

        if torch.cuda.is_available():
            if not self.centroid.is_cuda:
                self.centroid = self.centroid.cuda()
            if not self.sigma_inv.is_cuda:
                self.sigma_inv = self.sigma_inv.cuda()

        # compute MD given centroids and inverse covariance matrix
        dists = mahalanobis_distance_with_known_centroids_sigma_inv(
            self.centroid,
            None,
            self.sigma_inv,
            embeddings,
        )[:, 0]

        if self.max < dists.max():
            self.max = dists.max()
            if self.parameters_path is not None:
                torch.save(self.max, f"{self.full_path}/max.pt")
        if self.min > dists.min():
            self.min = dists.min()
            if self.parameters_path is not None:
                torch.save(self.min, f"{self.full_path}/min.pt")

        # norlmalise if required
        if self.normalize:
            dists = torch.clip((self.max - dists) / (self.max - self.min), min=0, max=1)

        return dists.cpu().detach().numpy()
