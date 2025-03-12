import numpy as np

import itertools
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch
from tqdm import tqdm

softmax = nn.Softmax(dim=1)


class GreedySemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self, nli_model):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "greedy_semantic_matrix_forward",
            "greedy_semantic_matrix_backward",
            "greedy_semantic_matrix",
        ], ["greedy_texts", "sample_texts"]

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

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E_f = []
        E_b = []
        E = []

        for i, pairs in enumerate(batch_pairs):
            dl = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
            probs_f = []
            probs_b = []

            for first_texts, second_texts in tqdm(dl):
                batch = list(zip(first_texts, second_texts))
                encoded = tokenizer.batch_encode_plus(
                    batch, padding=True, return_tensors="pt"
                ).to(device)
                logits = deberta.deberta(**encoded).logits.detach().to(device)
                probs_f.append(softmax(logits).cpu().detach())

                batch = list(zip(second_texts, first_texts))
                encoded = tokenizer.batch_encode_plus(
                    batch, padding=True, return_tensors="pt"
                ).to(device)
                logits = deberta.deberta(**encoded).logits.detach().to(device)
                probs_b.append(softmax(logits).cpu().detach())

            probs_f = torch.cat(probs_f, dim=0)
            probs_b = torch.cat(probs_b, dim=0)

            inv = batch_invs[i]

            entail_probs_f = probs_f[:, ent_id]
            entail_probs_b = probs_b[:, ent_id]

            E_f.append(entail_probs_f[inv].numpy())
            E_b.append(entail_probs_b[inv].numpy())
            E.append((entail_probs_f[inv].numpy() + entail_probs_b[inv].numpy()) / 2)

        E_f = np.stack(E_f)
        E_b = np.stack(E_b)
        E = np.stack(E)

        return {
            "greedy_semantic_matrix_forward": E_f,
            "greedy_semantic_matrix_backward": E_b,
            "greedy_semantic_matrix": E,
        }
