import numpy as np

from typing import Dict

from .estimator import Estimator
import sklearn.metrics


class PredictiveKernelEntropy(Estimator):
    """
    Predictive Kernel Entropy (PKE) uncertainty estimator.

    For each input, n answers are sampled and encoded with an external sentence
    embedding model. Uncertainty is the negative mean pairwise kernel similarity:

        PKE = -1 / (n*(n-1)) * sum_{i != j} k(X_i, X_j)

    Higher values indicate higher uncertainty.

    Reference: https://arxiv.org/abs/2310.05833

    Parameters:
        kernel (str): Kernel function — one of "rbf", "laplacian", "cosine". Default "rbf".
        gamma (float): Gamma parameter for the RBF and the laplacian kernels. Ignored for the cosine kernel.
    """

    def __init__(self, kernel: str = "rbf", gamma: float = 1.0):
        super().__init__(["sample_sentence_embeddings"], "sequence")
        if kernel not in ("rbf", "laplacian", "cosine"):
            raise ValueError(
                f"Unknown kernel '{kernel}'. Choose from: rbf, laplacian, cosine."
            )
        self.kernel = kernel
        self.gamma = gamma
        if kernel == "rbf":
            self.kernel_function = lambda x: sklearn.metrics.pairwise.rbf_kernel(
                x, gamma=gamma
            )
        elif kernel == "laplacian":
            self.kernel_function = lambda x: sklearn.metrics.pairwise.laplacian_kernel(
                x, gamma=gamma
            )
        elif kernel == "cosine":
            self.kernel_function = sklearn.metrics.pairwise.linear_kernel

    def __str__(self) -> str:
        return f"PredictiveKernelEntropy(kernel={self.kernel}, gamma={self.gamma})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        scores = []
        for embeddings in stats["sample_sentence_embeddings"]:
            E = np.array(embeddings)  # (n, d)
            n = len(E)
            if n < 2:
                scores.append(0.0)
                continue
            K = self.kernel_function(E)  # (n, n)
            scores.append((K.diagonal().sum() - K.sum()) / (n * (n - 1)))
        return np.array(scores)
