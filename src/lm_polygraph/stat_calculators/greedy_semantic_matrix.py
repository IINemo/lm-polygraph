import numpy as np

import itertools
from typing import Dict, List

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
        super().__init__(
            [
                "greedy_semantic_matrix_forward",
                "greedy_semantic_matrix_backward",
                "greedy_semantic_matrix",
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
        neutral_id = deberta.deberta.config.label2id["NEUTRAL"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E_f = []
        E_b = []
        E = []
        N_f = []
        N_b = []
        N = []
        C_f = []
        C_b = []
        C = []

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
            contra_probs_f = probs_f[:, contra_id]
            contra_probs_b = probs_b[:, contra_id]
            neutral_probs_f = probs_f[:, neutral_id]
            neutral_probs_b = probs_b[:, neutral_id]

            E_f.append(entail_probs_f[inv].numpy())
            E_b.append(entail_probs_b[inv].numpy())
            E.append((entail_probs_f[inv].numpy() + entail_probs_b[inv].numpy()) / 2)
            N_f.append(neutral_probs_f[inv].numpy())
            N_b.append(neutral_probs_b[inv].numpy())
            N.append((neutral_probs_f[inv].numpy() + neutral_probs_b[inv].numpy()) / 2)
            C_f.append(contra_probs_f[inv].numpy())
            C_b.append(contra_probs_b[inv].numpy())
            C.append((contra_probs_f[inv].numpy() + contra_probs_b[inv].numpy()) / 2)

        E_f = np.stack(E_f)
        E_b = np.stack(E_b)
        E = np.stack(E)
        N_f = np.stack(N_f)
        N_b = np.stack(N_b)
        N = np.stack(N)
        C_f = np.stack(C_f)
        C_b = np.stack(C_b)
        C = np.stack(C)

        return {
            "greedy_semantic_matrix_forward": E_f,
            "greedy_semantic_matrix_backward": E_b,
            "greedy_semantic_matrix": E,
            "greedy_semantic_matrix_neutral_forward": N_f,
            "greedy_semantic_matrix_neutral_backward": N_b,
            "greedy_semantic_matrix_neutral": N,
            "greedy_semantic_matrix_contra_forward": C_f,
            "greedy_semantic_matrix_contra_backward": C_b,
            "greedy_semantic_matrix_contra": C,
        }


class ConcatGreedySemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self, nli_model):
        super().__init__(
            [
                "concat_greedy_semantic_matrix_forward",
                "concat_greedy_semantic_matrix_backward",
                "concat_greedy_semantic_matrix",
            ],
            ["greedy_texts", "no_fewshot_input_texts", "sample_texts"],
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
        input_texts = dependencies["no_fewshot_input_texts"]


        batch_pairs = []
        batch_invs = []
        for texts, greedy_text, input_text in zip(batch_texts, batch_greedy_texts, input_texts):
            texts = [input_text + text for text in texts]
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(list(itertools.product([input_text + greedy_text], unique_texts)))
            batch_invs.append(inv)

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
        neutral_id = deberta.deberta.config.label2id["NEUTRAL"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E_f = []
        E_b = []
        E = []
        N_f = []
        N_b = []
        N = []
        C_f = []
        C_b = []
        C = []
        
        with torch.no_grad():
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
                contra_probs_f = probs_f[:, contra_id]
                contra_probs_b = probs_b[:, contra_id]
                neutral_probs_f = probs_f[:, neutral_id]
                neutral_probs_b = probs_b[:, neutral_id]

                E_f.append(entail_probs_f[inv].numpy())
                E_b.append(entail_probs_b[inv].numpy())
                E.append((entail_probs_f[inv].numpy() + entail_probs_b[inv].numpy()) / 2)
                N_f.append(neutral_probs_f[inv].numpy())
                N_b.append(neutral_probs_b[inv].numpy())
                N.append((neutral_probs_f[inv].numpy() + neutral_probs_b[inv].numpy()) / 2)
                C_f.append(contra_probs_f[inv].numpy())
                C_b.append(contra_probs_b[inv].numpy())
                C.append((contra_probs_f[inv].numpy() + contra_probs_b[inv].numpy()) / 2)

        E_f = np.stack(E_f)
        E_b = np.stack(E_b)
        E = np.stack(E)
        N_f = np.stack(N_f)
        N_b = np.stack(N_b)
        N = np.stack(N)
        C_f = np.stack(C_f)
        C_b = np.stack(C_b)
        C = np.stack(C)

        return {
            "concat_greedy_semantic_matrix_forward": E_f,
            "concat_greedy_semantic_matrix_backward": E_b,
            "concat_greedy_semantic_matrix": E,
            "concat_greedy_semantic_matrix_neutral_forward": N_f,
            "concat_greedy_semantic_matrix_neutral_backward": N_b,
            "concat_greedy_semantic_matrix_neutral": N,
            "concat_greedy_semantic_matrix_contra_forward": C_f,
            "concat_greedy_semantic_matrix_contra_backward": C_b,
            "concat_greedy_semantic_matrix_contra": C,
        }
