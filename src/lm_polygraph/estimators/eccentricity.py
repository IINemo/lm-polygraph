import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from .common import DEBERTA, compute_sim_score
from scipy.linalg import eigh
import torch.nn as nn

softmax = nn.Softmax(dim=1)


class Eccentricity(Estimator):
    def __init__(
            self,
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
            batch_size: int = 10,
            verbose: bool = False
    ):
        """
        It is a frobenious norm (euclidian norm) between all eigenvectors that are informative embeddings of graph Laplacian (lower this value -> answers are closer in terms of euclidian distance between embeddings = eigenvectors or higher = bigger uncertainty).

        Parameters:
            similarity_score (str): The argument to be processed. Possible values are:
                - 'NLI_score': As a similarity score NLI score is used.
                - 'Jaccard_score': As a similarity Jaccard score between responces is used.
            affinity (str): The argument to be processed. Possible values are. Relevant for the case of NLI similarity score:
                - 'entail': similarity(response_1, response_2) = p_entail(response_1, response_2)
                - 'contra': similarity(response_1, response_2) = 1 - p_contra(response_1, response_2)
        """
        if similarity_score == 'NLI_score':
            DEBERTA.setup()
            if affinity == 'entail':
                super().__init__(['semantic_matrix_entail',
                                  'blackbox_sample_texts'], 'sequence')
            else:
                super().__init__(['semantic_matrix_contra',
                                  'blackbox_sample_texts'], 'sequence')

        self.similarity_score = similarity_score
        self.batch_size = batch_size
        self.affinity = affinity
        self.verbose = verbose

    def __str__(self):
        if self.similarity_score == 'NLI_score':
            return f'Eccentricity_{self.similarity_score}_{self.affinity}'
        return f'Eccentricity_{self.similarity_score}'

    def U_Eccentricity(self, i, stats, k=2):
        answers = stats['blackbox_sample_texts']

        if self.similarity_score == 'NLI_score':
            if self.affinity == 'entail':
                W = stats['semantic_matrix_entail'][i, :, :]
            else:
                W = 1 - stats['semantic_matrix_contra'][i, :, :]
            W = (W + np.transpose(W)) / 2
        else:
            W = compute_sim_score(answers = answers, affinity = self.affinity, similarity_score = self.similarity_score)

        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt

        # k is hyperparameter  - Number of smallest eigenvectors to retrieve
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)
        smallest_eigenvectors = eigenvectors[:, :k]
        V_mat = smallest_eigenvectors - smallest_eigenvectors.mean(axis=0)

        norms = np.linalg.norm(V_mat, ord=2, axis=0)
        U_Ecc = np.linalg.norm(norms, 2)
        C_Ecc_s_j = norms
        return U_Ecc, C_Ecc_s_j

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for i, answers in enumerate(stats['blackbox_sample_texts']):
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_Eccentricity(i, stats)[0])
        return np.array(res)
