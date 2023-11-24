import os
import numpy as np
import torch

from typing import Dict
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from sklearn.covariance import MinCovDet

from .estimator import Estimator

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-15, 0, 1)]


def save_array(array, filename):
    with open(filename, "wb") as f:
        np.save(f, array)


def load_array(filename):
    with open(filename, "rb") as f:
        array = np.load(f)
    return array


def MCD_covariance(X, y=None, label=None, seed=42):
    try:
        if label is None:
            cov = MinCovDet(random_state=seed).fit(X)
        else:
            cov = MinCovDet(random_state=seed).fit(X[y == label])
    except ValueError:
        print(
            "****************Try fitting covariance with support_fraction=0.9 **************"
        )
        try:
            if label is None:
                cov = MinCovDet(random_state=seed, support_fraction=0.9).fit(X)
            else:
                cov = MinCovDet(random_state=seed, support_fraction=0.9).fit(
                    X[y == label]
                )
        except ValueError:
            print(
                "****************Try fitting covariance with support_fraction=1.0 **************"
            )
            if label is None:
                cov = MinCovDet(random_state=seed, support_fraction=1.0).fit(X)
            else:
                cov = MinCovDet(random_state=seed, support_fraction=1.0).fit(
                    X[y == label]
                )
    return cov


class RDESeq(Estimator):
    """
    The RDE method improves over MD by reducing the dimensionality of h(x) via PCA decomposition.
    It also computes the covariance matrix in a robust way using the Minimum Covariance Determinant
    estimate (Rousseeuw, 1984).
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
    ):
        super().__init__(["embeddings", "train_embeddings"], "sequence")
        self.pca = None
        self.MCD = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.is_fitted = False

        if self.parameters_path is not None:
            self.full_path = f"{self.parameters_path}/rde_{self.embeddings_type}"
            os.makedirs(self.full_path, exist_ok=True)
            if os.path.exists(f"{self.full_path}/covariance.npy"):
                self.pca = self.load_pca()
                self.MCD = self.load_mcd()
                self.max = load_array(f"{self.full_path}/max.npy")
                self.min = load_array(f"{self.full_path}/min.npy")
                self.is_fitted = True

    def __str__(self):
        return f"RDESeq_{self.embeddings_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # take embeddings
        embeddings = stats[f"embeddings_{self.embeddings_type}"]

        # define PCA with rbf kernel and n_components equal 100
        if not self.is_fitted:
            self.pca = KernelPCA(
                n_components=100, kernel="rbf", random_state=42, gamma=None
            )
            X_pca_train = self.pca.fit_transform(
                stats[f"train_embeddings_{self.embeddings_type}"]
            )
            if self.parameters_path is not None:
                self.save_pca()

        # define mean covariance distance
        if not self.is_fitted:
            self.MCD = MCD_covariance(X_pca_train)
            if self.parameters_path is not None:
                self.save_mcd()
            self.is_fitted = True

        # transform test data based on pca
        X_pca_test = self.pca.transform(embeddings)

        # compute MD in space of reduced dimensionality
        dists = self.MCD.mahalanobis(X_pca_test)

        if self.max < dists.max():
            self.max = dists.max()
            if self.parameters_path is not None:
                save_array(self.max, f"{self.full_path}/max.npy")
        if self.min > dists.min():
            self.min = dists.min()
            if self.parameters_path is not None:
                save_array(self.min, f"{self.full_path}/min.npy")

        if self.normalize:
            dists = np.clip(
                (self.max - dists) / (self.max - self.min), a_min=0, a_max=1
            )

        return dists

    def save_mcd(self):
        save_array(self.MCD.covariance_, f"{self.full_path}/covariance.npy")
        save_array(self.MCD.location_, f"{self.full_path}/location.npy")
        save_array(self.MCD.precision_, f"{self.full_path}/precision.npy")

    def save_pca(self):
        save_array(self.pca.eigenvalues_, f"{self.full_path}/eigenvalues.npy")
        save_array(self.pca.eigenvectors_, f"{self.full_path}/eigenvectors.npy")
        save_array(self.pca.X_fit_, f"{self.full_path}/X_fit.npy")
        save_array(self.pca._centerer.K_fit_rows_, f"{self.full_path}/K_fit_rows.npy")
        save_array(self.pca._centerer.K_fit_all_, f"{self.full_path}/K_fit_all.npy")

    def load_mcd(self):
        self.MCD = MinCovDet(random_state=42)
        self.MCD.covariance_ = load_array(f"{self.full_path}/covariance.npy")
        self.MCD.location_ = load_array(f"{self.full_path}/location.npy")
        self.MCD.precision_ = load_array(f"{self.full_path}/precision.npy")
        return self.MCD

    def load_pca(self):
        self.pca = KernelPCA(
            n_components=100, kernel="rbf", random_state=42, gamma=None
        )
        self.pca._centerer = KernelCenterer()
        self.pca.eigenvalues_ = load_array(f"{self.full_path}/eigenvalues.npy")
        self.pca.eigenvectors_ = load_array(f"{self.full_path}/eigenvectors.npy")
        self.pca.X_fit_ = load_array(f"{self.full_path}/X_fit.npy")
        self.pca._centerer.K_fit_rows_ = load_array(f"{self.full_path}/K_fit_rows.npy")
        self.pca._centerer.K_fit_all_ = load_array(f"{self.full_path}/K_fit_all.npy")
        self.pca.gamma_ = None
        return self.pca
