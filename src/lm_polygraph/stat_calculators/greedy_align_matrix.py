import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch
from tqdm import tqdm


class GreedyAlignMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self, scorer):
        super().__init__(
            [
                "greedy_align_semantic_matrix_forward",
                "greedy_align_semantic_matrix_backward",
                "greedy_align_semantic_matrix",
            ],
            ["greedy_texts", "sample_texts"],
        )
        self.scorer = scorer

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
            texts = [text if text.strip() != "" else "<empty>" for text in texts]
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([greedy_text], unique_texts)))
            batch_invs.append(inv)

        E_f = []
        E_b = []
        E = []

        for i, pairs in tqdm(enumerate(batch_pairs)):
            sim_mat_f = []
            sim_mat_b = []
            first_texts, second_texts = zip(*pairs)
            sim_mat_f = np.array(self.scorer.score(claims=first_texts, contexts=second_texts))
            sim_mat_b = np.array(self.scorer.score(claims=second_texts, contexts=first_texts))

            inv = batch_invs[i]

            E_f.append(sim_mat_f[inv])
            E_b.append(sim_mat_b[inv])
            E.append((sim_mat_f[inv] + sim_mat_b[inv]) / 2)


        E_f = np.stack(E_f)
        E_b = np.stack(E_b)
        E = np.stack(E)

        return {
            "greedy_align_semantic_matrix_forward": E_f,
            "greedy_align_semantic_matrix_backward": E_b,
            "greedy_align_semantic_matrix": E,
        }
