import numpy as np
import scipy.linalg

from typing import Dict

from .estimator import Estimator


def heat_kernel(laplacian: np.ndarray, t: float) -> np.ndarray:
    return scipy.linalg.expm(-t * laplacian)


def normalize_kernel(K: np.ndarray) -> np.ndarray:
    EPS = 1e-12
    diagonal_values = np.sqrt(np.diag(K)) + EPS
    normalized_kernel = K / np.outer(diagonal_values, diagonal_values)
    return normalized_kernel


def scale_entropy(entropy: np.ndarray, n_classes: int) -> np.ndarray:
    max_entropy = -np.log(
        1.0 / n_classes
    )  # For a discrete distribution with num_classes
    scaled_entropy = entropy / max_entropy
    return scaled_entropy


def vn_entropy(K: np.ndarray, normalize=True, scale=True, jitter=0) -> np.float64:
    if normalize:
        K = normalize_kernel(K) / K.shape[0]
    result = 0
    eigvs = np.linalg.eig(K + jitter * np.eye(K.shape[0])).eigenvalues.astype(
        np.float64
    )
    for e in eigvs:
        if np.abs(e) > 1e-8:
            result -= e * np.log(e)
    if scale:
        result = scale_entropy(result, K.shape[0])
    return np.float64(result)


class KernelLanguageEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Kernel Language Entropy" as provided in the paper https://arxiv.org/pdf/2405.20003
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    This method calculates KLE(Kheat) = VNE(Kheat), where VNE is von Neumann entropy and
    Kheat is a heat kernel of a semantic graph over language model's outputs.
    """

    def __init__(self, t: float = 0.3):  # Default value is taken from the paper
        super().__init__(
            ["semantic_matrix_entail", "semantic_matrix_contra"], "sequence"
        )
        self.t = t

    def __str__(self):
        return "KernelLanguageEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates KLE(Kheat) uncertainty of a language model.
        1. Let S1, ..., Sn be a set of LLM generations.
        2. Let NLI'(Si, Sj) = one-hot prediction over (entailment, neutral class, contradiction)
        Note that NLI'(Si, Sj) is calculated in stats
        3. Let W be a matrix, such that Wij = wNLI'(Si, Sj), where w = (1, 0.5, 0)
        4. Let D be a matrix, such that Dii = sum(Wij) over j.
        4. Let L be a laplacian matrix of W, i.e. L = W - D.
        5. Let Kheat = heat kernel of W, i.e. Kheat = expm(-t * L), where t is a hyperparameter.
        6. Finally, KLE(x) = VNE(Kheat), where VNE(A) = -Tr(A log A).
        """
        weighted_graph = stats["semantic_matrix_entail"] + 0.5 * (
            np.ones(stats["semantic_matrix_entail"].shape)
            - stats["semantic_matrix_entail"]
            - stats["semantic_matrix_contra"]
        )
        degrees = np.diag(np.sum(weighted_graph, axis=0))
        laplacian = weighted_graph - degrees
        heat_kernels = heat_kernel(laplacian, self.t)
        return [vn_entropy(heat_kernel) for heat_kernel in heat_kernels]
