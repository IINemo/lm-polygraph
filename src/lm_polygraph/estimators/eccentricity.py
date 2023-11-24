import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from .common import DEBERTA, compute_sim_score
from scipy.linalg import eigh


class Eccentricity(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Eccentricity" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    Method calculates a frobenious (euclidian) norm between all eigenvectors that are informative embeddings
    of graph Laplacian (lower norm -> closer embeddings -> higher eigenvectors -> greater uncertainty).
    """

    def __init__(
        self,
        similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
        affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
        verbose: bool = False,
        thres: float = 0.9,
    ):
        """
        See parameters descriptions in https://arxiv.org/abs/2305.19187.
        Parameters:
            similarity_score (str): similarity score for matrix construction. Possible values:
                - 'NLI_score': Natural Language Inference similarity
                - 'Jaccard_score': Jaccard score similarity
            affinity (str): affinity method, relevant only when similarity_score='NLI_score'. Possible values:
                - 'entail': similarity(response_1, response_2) = p_entail(response_1, response_2)
                - 'contra': similarity(response_1, response_2) = 1 - p_contra(response_1, response_2)
        """
        if similarity_score == "NLI_score":
            DEBERTA.setup()
            if affinity == "entail":
                super().__init__(
                    ["semantic_matrix_entail", "blackbox_sample_texts"], "sequence"
                )
            else:
                super().__init__(
                    ["semantic_matrix_contra", "blackbox_sample_texts"], "sequence"
                )
        else:
            super().__init__(["blackbox_sample_texts"], "sequence")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose
        self.thres = thres

    def __str__(self):
        if self.similarity_score == "NLI_score":
            return f"Eccentricity_{self.similarity_score}_{self.affinity}"
        return f"Eccentricity_{self.similarity_score}"

    def U_Eccentricity(self, i, stats):
        answers = stats["blackbox_sample_texts"][i]

        if self.similarity_score == "NLI_score":
            if self.affinity == "entail":
                W = stats["semantic_matrix_entail"][i, :, :]
            else:
                W = 1 - stats["semantic_matrix_contra"][i, :, :]
            W = (W + np.transpose(W)) / 2
        else:
            W = compute_sim_score(
                answers=answers,
                affinity=self.affinity,
                similarity_score=self.similarity_score,
            )

        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt

        # k is hyperparameter  - Number of smallest eigenvectors to retrieve
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)

        if self.thres is not None:
            keep_mask = eigenvalues < self.thres
            eigenvalues, smallest_eigenvectors = (
                eigenvalues[keep_mask],
                eigenvectors[:, keep_mask],
            )

        smallest_eigenvectors = smallest_eigenvectors.T

        C_Ecc_s_j = (-1) * np.asarray(
            [np.linalg.norm(x - x.mean(0), 2) for x in smallest_eigenvectors]
        )
        U_Ecc = np.linalg.norm(C_Ecc_s_j, 2)

        return U_Ecc, C_Ecc_s_j

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'blackbox_sample_texts',
                * matrix with semantic similarities in 'semantic_matrix_entail'/'semantic_matrix_contra'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        res = []
        for i, answers in enumerate(stats["blackbox_sample_texts"]):
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_Eccentricity(i, stats)[0])
        return np.array(res)
