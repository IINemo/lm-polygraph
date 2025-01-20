import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)


class GreedySemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self, nli_model):
        super().__init__(
            [
                "greedy_semantic_matrix_entail",
                "greedy_semantic_matrix_contra",
            ],
            ["greedy_texts", "sample_texts"],
        )
        self.is_deberta_setup = False
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size

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

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E = []
        C = []

        for i, pairs in enumerate(batch_pairs):
            dl = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
            probs = []
            for first_texts, second_texts in dl:
                batch = list(zip(first_texts, second_texts))
                encoded = tokenizer.batch_encode_plus(
                    batch, padding=True, return_tensors="pt"
                ).to(device)
                logits = deberta.deberta(**encoded).logits.detach().to(device)
                probs.append(softmax(logits).cpu().detach())
            probs = torch.cat(probs, dim=0)

            inv = batch_invs[i]

            entail_probs = probs[:, ent_id]
            contra_probs = probs[:, contra_id]

            E.append(entail_probs[inv].numpy())
            C.append(contra_probs[inv].numpy())

        E = np.stack(E)
        C = np.stack(C)

        return {
            "greedy_semantic_matrix_entail": E,
            "greedy_semantic_matrix_contra": C,
        }
