import numpy as np
import logging
from typing import Dict, Literal
from .estimator import Estimator
from .common import compute_sim_score
from scipy.linalg import eigh

log = logging.getLogger(__name__)


class Eccentricity(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Eccentricity" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.model_adapters.blackbox_model.BlackboxModel/WhiteboxModel).

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
            if affinity == "entail":
                super().__init__(["semantic_matrix_entail", "sample_texts"], "sequence")
            else:
                super().__init__(["semantic_matrix_contra", "sample_texts"], "sequence")
        else:
            super().__init__(["sample_texts"], "sequence")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose
        self.thres = thres

    def __str__(self):
        if self.similarity_score == "NLI_score":
            return f"Eccentricity_{self.similarity_score}_{self.affinity}"
        return f"Eccentricity_{self.similarity_score}"

    def U_Eccentricity(self, i, stats):
        answers = stats["sample_texts"][i]

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

        ave_embedding = np.mean(smallest_eigenvectors, 0)
        deviations = smallest_eigenvectors - ave_embedding

        U_Ecc = np.linalg.norm(deviations, ord='fro')

        return U_Ecc

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * matrix with semantic similarities in 'semantic_matrix_entail'/'semantic_matrix_contra'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        res = []
        for i, answers in enumerate(stats["sample_texts"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            res.append(self.U_Eccentricity(i, stats))
        return np.array(res)


class CEccentricity(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Eccentricity" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.model_adapters.blackbox_model.BlackboxModel/WhiteboxModel).

    Method calculates a frobenious (euclidian) norm between all eigenvectors that are informative embeddings
    of graph Laplacian (lower norm -> closer embeddings -> higher eigenvectors -> greater uncertainty).
    """

    def __init__(
        self,
        similarity_score: str = "NLI_score",
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
            if affinity == "entail":
                super().__init__(["greedy_semantic_matrix_entail", "semantic_matrix_entail", "sample_texts"], "sequence")
            else:
                super().__init__(["greedy_semantic_matrix_contra", "semantic_matrix_contra", "sample_texts"], "sequence")
        else:
            raise ValueError("CEccentricity only supports NLI_score as similarity_score.")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose
        self.thres = thres

    def __str__(self):
        return f"Eccentricity_{self.similarity_score}_{self.affinity}"

    def C_Eccentricity(self, i, stats):
        answers = stats["sample_texts"][i]

        if self.affinity == "entail":
            gW = stats["greedy_semantic_matrix_entail"][i, :]
            W = stats["semantic_matrix_entail"][i, :, :]
        else:
            gW = stats["greedy_semantic_matrix_contra"][i, :]
            W = stats["semantic_matrix_contra"][i, :, :]

        # Add gW to W as both first row and first column, to account for greedy answer. Setting self-similarity to 1.
        W = np.insert(W, 0, gW, axis=0)
        W = np.insert(W, 0, np.append(1, gW), axis=1)

        if self.affinity == "contra":
            W = 1 - W

        W = (W + np.transpose(W)) / 2

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

        ave_embedding = np.mean(smallest_eigenvectors, 0)
        deviations = smallest_eigenvectors - ave_embedding

        greedy_deviation = smallest_eigenvectors[0, :]

        C_Ecc = np.linalg.norm(greedy_deviation, ord=2)

        return C_Ecc

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the uncertainties for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * matrix with semantic similarities in 'semantic_matrix_entail'/'semantic_matrix_contra'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        res = []
        for i, answers in enumerate(stats["sample_texts"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            res.append(self.C_Eccentricity(i, stats))
        return np.array(res)
