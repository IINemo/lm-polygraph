import torch
import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from .common import DEBERTA, compute_sim_score
import torch.nn as nn


class DegMat(Estimator):
    def __init__(
            self,
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
            batch_size: int = 10,
            verbose: bool = False
    ):
        """
        Elements on diagonal of matrix D are sums of similarities between the particular number (position in matrix) and other answers. Thus, it is an average pairwise distance (less = more confident because distance between answers is smaller or higher = bigger uncertainty).

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
        else:
            super().__init__(['blackbox_sample_texts'], 'sequence')

        self.similarity_score = similarity_score
        self.batch_size = batch_size
        self.affinity = affinity
        self.verbose = verbose

    def __str__(self):
        if self.similarity_score == 'NLI_score':
            return f'DegMat_{self.similarity_score}_{self.affinity}'
        return f'DegMat_{self.similarity_score}'

    def U_DegMat(self, i, stats):
        # The Degree Matrix
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
        return np.trace(len(answers) - D) / (len(answers) ** 2)

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for i, answers in enumerate(stats['blackbox_sample_texts']):
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_DegMat(i, stats))
        return np.array(res)
