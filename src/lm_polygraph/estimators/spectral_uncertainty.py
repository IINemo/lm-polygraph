import numpy as np

from typing import Dict

from .estimator import Estimator
import sklearn.metrics
import scipy


class SpectralUncertainty(Estimator):
    """
    Spectral Uncertainty estimator.

    For each input, n answers are sampled and encoded with an external sentence
    embedding model. The empirical kernel matrix K is built from pairwise kernel
    values between embeddings, the kernel matrix is normalized to unit trace, and
    as a result, eigenvalues of K can be interpreted as a probability distribution
    over latent semantic directions. Finally, uncertainty is computed as the Shannon
    entropy of that distribution:

        H = -Σ λᵢ log(λᵢ)

    Higher values indicate higher uncertainty (more spread in the spectrum).

    Reference: https://arxiv.org/abs/2509.22272

    Parameters:
        kernel (str): Kernel function — one of "rbf", "laplacian", "cosine". Default "rbf".
        gamma (float): Gamma parameter for rbf/laplacian kernels. Ignored for cosine.
        eps (float): Small value to be added to the diagonal to avoid numerical issues.
    """

    def __init__(self, kernel: str = "rbf", gamma: float = 1.0, eps: float = 1e-6):
        super().__init__(["sample_sentence_embeddings"], "sequence")
        if kernel not in ("rbf", "laplacian", "cosine"):
            raise ValueError(
                f"Unknown kernel '{kernel}'. Choose from: rbf, laplacian, cosine."
            )
        self.kernel = kernel
        self.gamma = gamma
        self.eps = eps
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
        return f"SpectralUncertainty(kernel={self.kernel}, gamma={self.gamma})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        scores = []
        for embeddings in stats["sample_sentence_embeddings"]:
            E = np.array(embeddings)  # (n, d)
            n = len(E)
            if n < 2:
                scores.append(0.0)
                continue
            K = self.kernel_function(E)  # (n, n) symmetric
            K /= n  # Normalizing to trace =1. Here we assume that the embeddings are normalized, so the trace is n.
            try:
                eigenvalues, _ = scipy.linalg.eigh(K)
            except np.linalg.LinAlgError:
                print(
                    f"Linalg error happened because of condition {np.linalg.cond(K)}. Trying numpy function."
                )
                try:
                    eigenvalues, _ = np.linalg.eigh(K)
                except np.linalg.LinAlgError:
                    epsilon = 1e-6  # You can tune this
                    K += epsilon * np.eye(K.shape[0])
                    print(
                        f"Numy function failed. Adding {epsilon} to the diagonal. New condition is {np.linalg.cond(K)}"
                    )
                    eigenvalues, _ = scipy.linalg.eigh(K)
            # Due to numerical issues, sometimes negative eigenvalues appear (although theoretically impossible because PSD). These are very marginal and are clipped to 0.
            eigenvalues = np.array([(ev if ev >= 0 else 0.0) for ev in eigenvalues])
            H = scipy.stats.entropy(eigenvalues)
            scores.append(H)
        return np.array(scores)
