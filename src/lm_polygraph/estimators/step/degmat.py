import numpy as np
import logging
from typing import Dict, Literal
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.common import compute_sim_score
from lm_polygraph.stat_calculators.step.utils import flatten, reconstruct

log = logging.getLogger(__name__)


class StepsDegMat(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "The Degree Matrix" as provided in the paper https://arxiv.org/abs/2305.19187.
    Works with both whitebox and blackbox models (initialized using
    lm_polygraph.utils.model.BlackboxModel/WhiteboxModel).

    Elements on diagonal of matrix D are sums of similarities between the particular number
    (position in matrix) and other answers. Thus, it is an average pairwise distance
    (lower values indicated smaller distance between answers which means greater uncertainty).
    """

    def __init__(
            self,
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
            verbose: bool = False,
    ):
        """
        Parameters:
            similarity_score (str): The argument to be processed. Possible values are:
                - 'NLI_score': As a similarity score NLI score is used.
                - 'Jaccard_score': As a similarity Jaccard score between responces is used.
            affinity (str): The argument to be processed. Possible values are. Relevant for the case of NLI similarity score:
                - 'entail': similarity(response_1, response_2) = p_entail(response_1, response_2)
                - 'contra': similarity(response_1, response_2) = 1 - p_contra(response_1, response_2)
        """
        if similarity_score == "NLI_score":
            if affinity == "entail":
                super().__init__(["steps_semantic_matrix_entail", "sample_steps_texts"], "claim")
            else:
                super().__init__(["steps_semantic_matrix_contra", "sample_steps_texts"], "claim")
        else:
            super().__init__(["sample_steps_texts"], "claim")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose

    def __str__(self):
        if self.similarity_score == "NLI_score":
            return f"StepsDegMat_{self.similarity_score}_{self.affinity}"
        return f"StepsDegMat_{self.similarity_score}"

    def U_DegMat(self, sample_texts, semantic_matrix_entail, semantic_matrix_contra):
        # The Degree Matrix
        answers = sample_texts

        if self.similarity_score == "NLI_score":
            if self.affinity == "entail":
                W = semantic_matrix_entail[:, :]
            else:
                W = 1 - semantic_matrix_contra[:, :]
            W = (W + np.transpose(W)) / 2
        else:
            W = compute_sim_score(
                answers=answers,
                affinity=self.affinity,
                similarity_score=self.similarity_score,
            )

        D = np.diag(W.sum(axis=1))
        return np.trace(len(answers) - D) / (len(answers) ** 2)

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
        sample_steps_texts = flatten(stats["sample_steps_texts"])
        semantic_matrix_entail = flatten(stats["steps_semantic_matrix_entail"])
        semantic_matrix_contra = flatten(stats["steps_semantic_matrix_contra"])
        for i in range(len(sample_steps_texts)):
            res.append(self.U_DegMat(
                sample_steps_texts[i],
                semantic_matrix_entail[i],
                semantic_matrix_contra[i],
            ))
        return reconstruct(res, stats["sample_steps_texts"])
