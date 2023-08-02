import numpy as np

from typing import Dict, Literal

from .common import DEBERTA, compute_sim_score
from .estimator import Estimator
import torch.nn as nn

softmax = nn.Softmax(dim=1)


class EigValLaplacian(Estimator):
    def __init__(
            self,
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            order: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
            batch_size: int = 10,
            verbose: bool = False,
            epsilon: float = 1e-13
    ):
        """
        (Due to the Theorem) A continuous analogue to the number of semantic sets (higher = bigger uncertainty).

        Parameters:
            similarity_score (str): The argument to be processed. Possible values are:
                - 'NLI_score': As a similarity score NLI score is used.
                - 'Jaccard_score': As a similarity Jaccard score between responces is used.
            order (str): The argument to be processed. Possible values are. Relevant for the case of NLI similarity score:
                - 'forward': Compute entailment probability between response_1 and response_2 as p(response_1 -> response_2).
                - 'backward': Compute entailment probability between response_1 and response_2 as p(response_2 -> response_1).
        """
        super().__init__(['sample_texts'], 'sequence')
        self.similarity_score = similarity_score
        self.batch_size = batch_size
        if self.similarity_score == "NLI_score":
            DEBERTA.setup()
        self.order = order
        self.verbose = verbose
        self.epsilon = epsilon

    def __str__(self):
        if self.similarity_score == 'NLI_score':
            return f'EigValLaplacian_{self.similarity_score}_{self.order}'
        return f'EigValLaplacian_{self.similarity_score}'

    def U_EigVal_Laplacian(self, answers):
        W = compute_sim_score(answers, self.order, self.epsilon, self.similarity_score)
        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt
        return sum([max(0, 1 - lambda_k) for lambda_k in np.linalg.eig(L)[0]])

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for answers in stats['sample_texts']:
            if self.verbose:
                print(f"generated answers: {answers}")
            res.append(self.U_EigVal_Laplacian(answers))
        return np.array(res)
