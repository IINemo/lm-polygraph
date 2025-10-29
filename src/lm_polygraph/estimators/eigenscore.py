import numpy as np
from scipy.linalg import eigh

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


class EigenScore(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "EigenScore" as provided in the paper https://openreview.net/forum?id=Zj12nzlQbz.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    Uses embeddings for the last generated token from the middle layer of the model.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
    ):
        super().__init__(["sample_embeddings"], "sequence")
        self.alpha = alpha
        self.J_d = None

    def __str__(self):
        return "EigenScore"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the EigenScore score for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                *  embeddings for several sampled texts in 'sample_embeddings'
        Returns:
            np.ndarray: float EigenScore score for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        sample_embeddings = stats["sample_embeddings"]
        ue = []
        for embeddings in sample_embeddings:
            sentence_embeddings = np.array(embeddings)
            if self.J_d is None:
                dim = sentence_embeddings.shape[-1]
                self.J_d = np.eye(dim) - 1 / dim * np.ones((dim, dim))
            covariance = sentence_embeddings @ self.J_d @ sentence_embeddings.T
            reg_covariance = covariance + self.alpha * np.eye(covariance.shape[0])
            eigenvalues = eigh(reg_covariance, eigvals_only=True)
            ue.append(
                np.mean(np.log([val if val > 0 else 1e-10 for val in eigenvalues]))
            )
        return np.array(ue)
