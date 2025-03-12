import numpy as np
from typing import Dict

from lm_polygraph.estimators.estimator import Estimator


class LUQ(Estimator):
    def __init__(self):
        super().__init__(
            ["semantic_matrix_entail_logits", "semantic_matrix_contra_logits"],
            "sequence",
        )

    def __str__(self):
        return "LUQ"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:

        entail_logits = stats["semantic_matrix_entail_logits"]
        contra_logits = stats["semantic_matrix_contra_logits"]

        luq = []
        for j in range(len(entail_logits)):
            sim_scores = np.exp(entail_logits[j]) / (
                np.exp(entail_logits[j]) + np.exp(contra_logits[j])
            )
            sim_scores = (sim_scores.sum(axis=1) - sim_scores.diagonal()) / (
                sim_scores.shape[-1] - 1
            )
            luq.append(1 - sim_scores.mean())

        return np.array(luq)
