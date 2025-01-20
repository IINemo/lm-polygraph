import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch
from rouge_score import rouge_scorer

class GreedyRougeLSemanticMatrixCalculator(StatCalculator):
    def __init__(self):
        super().__init__(
            [
                "greedy_semantic_matrix",
            ],
            ["greedy_texts", "sample_texts"],
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
        batch_greedy_texts = dependencies["greedy_texts"]

        batch_pairs = []
        batch_invs = []
        for texts, greedy_text in zip(batch_texts, batch_greedy_texts):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_text], unique_texts)))
            batch_invs.append(inv)


        E = []

        for i, pairs in enumerate(batch_pairs):
            sim_mat = []
            for first_texts, second_texts in pairs:
                sim_mat.append(self.scorer.score(first_texts, second_texts)['rougeL'].fmeasure)

            sim_mat = np.array(sim_mat)

            inv = batch_invs[i]
            E.append(sim_mat[inv])

        E = np.stack(E)

        return {
            "greedy_rouge_semantic_matrix": E,
        }
