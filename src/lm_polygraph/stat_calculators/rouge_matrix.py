import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch
from rouge_score import rouge_scorer


class RougeLSemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self):
        super().__init__(
            [
                "rouge_semantic_matrix",
            ],
            ["sample_texts"],
        )
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:

        batch_texts = dependencies["sample_texts"]

        batch_pairs = []
        batch_invs = []
        batch_counts = []
        for texts in batch_texts:
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product(unique_texts, unique_texts)))
            batch_invs.append(inv)
            batch_counts.append(len(unique_texts))

        E = []

        for i, pairs in enumerate(batch_pairs):
            sim_mat = []
            for first_texts, second_texts in pairs:
                sim_mat.append(self.scorer.score(first_texts, second_texts)['rougeL'].fmeasure)

            sim_mat = np.array(sim_mat)
            unique_mat_shape = (batch_counts[i], batch_counts[i])
            sim_mat = sim_mat.reshape(unique_mat_shape)

            inv = batch_invs[i]

            E.append(sim_mat[inv, :][:, inv])

        E = np.stack(E)

        return {
            "rouge_semantic_matrix": E,
        }
