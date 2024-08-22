import numpy as np

from typing import Dict

from .estimator import Estimator


class Linguistic1S(Estimator):
    """
    Estimates sequence-level uncertainty of a language model by extracting
    the confidence estimate from the model's answer using a provided regex.
    Model is expected to output confidence using one of the provided expressions,
    i.e. "Highly Likely", "Probable" etc. Mapping between expressions of
    confidence and their corresponding numerical values has to be provided.
    To use this estimator, model has to be correctly prompted to output
    it's confidence in the answer.
    Adapted from the original implementation in the paper https://arxiv.org/abs/2305.14975
    """

    def __init__(self, expressions: Dict[str, float], name_postfix=""):
        self.expressions = expressions
        self.postfix = name_postfix
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return f"Linguistic1S{self.postfix}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ues = []
        for answer in stats["greedy_texts"]:
            ue = np.nan
            for expression, confidence in self.expressions.items():
                if expression in answer:
                    ue = 1 - confidence
                    break
            ues.append(ue)

        return np.array(ues)
