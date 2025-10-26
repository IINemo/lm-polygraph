import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch
from tqdm import tqdm


class BaseGreedySemanticMatrixCalculator(StatCalculator):
    """
    Base class for calculating the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self, nli_model):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model

    def calculate_semantic_matrix(self, batch_pairs, batch_invs):
        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size

        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
        neutral_id = deberta.deberta.config.label2id["NEUTRAL"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E_f_tensors = []
        E_b_tensors = []
        E_tensors = []
        N_f_tensors = []
        N_b_tensors = []
        N_tensors = []
        C_f_tensors = []
        C_b_tensors = []
        C_tensors = []
        with torch.no_grad():
            for i, pairs in enumerate(batch_pairs):
                dl = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
                probs_f = []
                probs_b = []
                for first_texts, second_texts in tqdm(dl):
                    batch = list(zip(first_texts, second_texts))
                    encoded = tokenizer.batch_encode_plus(
                        batch, padding=True, return_tensors="pt"
                    ).to(deberta.device)
                    logits = deberta.deberta(**encoded).logits
                    probs_f.append(softmax(logits))

                    batch = list(zip(second_texts, first_texts))
                    encoded = tokenizer.batch_encode_plus(
                        batch, padding=True, return_tensors="pt"
                    ).to(deberta.device)
                    logits = deberta.deberta(**encoded).logits
                    probs_b.append(softmax(logits))

                probs_f = torch.cat(probs_f, dim=0)
                probs_b = torch.cat(probs_b, dim=0)

                inv = batch_invs[i]

                entail_probs_f = probs_f[:, ent_id][inv]
                entail_probs_b = probs_b[:, ent_id][inv]
                contra_probs_f = probs_f[:, contra_id][inv]
                contra_probs_b = probs_b[:, contra_id][inv]
                neutral_probs_f = probs_f[:, neutral_id][inv]
                neutral_probs_b = probs_b[:, neutral_id][inv]

                E_f_tensors.append(entail_probs_f)
                E_b_tensors.append(entail_probs_b)
                E_tensors.append((entail_probs_f + entail_probs_b) / 2)
                N_f_tensors.append(neutral_probs_f)
                N_b_tensors.append(neutral_probs_b)
                N_tensors.append((neutral_probs_f + neutral_probs_b) / 2)
                C_f_tensors.append(contra_probs_f)
                C_b_tensors.append(contra_probs_b)
                C_tensors.append((contra_probs_f + contra_probs_b) / 2)

            # Stack tensors and then convert to numpy arrays at the end
            E_f = torch.stack(E_f_tensors).cpu().numpy()
            E_b = torch.stack(E_b_tensors).cpu().numpy()
            E = torch.stack(E_tensors).cpu().numpy()
            N_f = torch.stack(N_f_tensors).cpu().numpy()
            N_b = torch.stack(N_b_tensors).cpu().numpy()
            N = torch.stack(N_tensors).cpu().numpy()
            C_f = torch.stack(C_f_tensors).cpu().numpy()
            C_b = torch.stack(C_b_tensors).cpu().numpy()
            C = torch.stack(C_tensors).cpu().numpy()

        return (E_f, E_b, E, N_f, N_b, N, C_f, C_b, C)


class GreedySemanticMatrixCalculator(BaseGreedySemanticMatrixCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    @staticmethod
    def meta_info():
        stats = [
            "greedy_semantic_matrix_entail_forward",
            "greedy_semantic_matrix_entail_backward",
            "greedy_semantic_matrix_entail",
            "greedy_semantic_matrix_neutral_forward",
            "greedy_semantic_matrix_neutral_backward",
            "greedy_semantic_matrix_neutral",
            "greedy_semantic_matrix_contra_forward",
            "greedy_semantic_matrix_contra_backward",
            "greedy_semantic_matrix_contra",
        ]
        dependencies = ["greedy_texts", "sample_texts"]
        return stats, dependencies

    def __init__(self, nli_model):
        super().__init__(nli_model)

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

        E_f, E_b, E, N_f, N_b, N, C_f, C_b, C = self.calculate_semantic_matrix(
            batch_pairs, batch_invs
        )

        return {
            "greedy_semantic_matrix_entail_forward": E_f,
            "greedy_semantic_matrix_entail_backward": E_b,
            "greedy_semantic_matrix_entail": E,
            "greedy_semantic_matrix_neutral_forward": N_f,
            "greedy_semantic_matrix_neutral_backward": N_b,
            "greedy_semantic_matrix_neutral": N,
            "greedy_semantic_matrix_contra_forward": C_f,
            "greedy_semantic_matrix_contra_backward": C_b,
            "greedy_semantic_matrix_contra": C,
        }


class ConcatGreedySemanticMatrixCalculator(BaseGreedySemanticMatrixCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    @staticmethod
    def meta_info():
        stats = [
            "concat_greedy_semantic_matrix_entail_forward",
            "concat_greedy_semantic_matrix_entail_backward",
            "concat_greedy_semantic_matrix_entail",
            "concat_greedy_semantic_matrix_neutral_forward",
            "concat_greedy_semantic_matrix_neutral_backward",
            "concat_greedy_semantic_matrix_neutral",
            "concat_greedy_semantic_matrix_contra_forward",
            "concat_greedy_semantic_matrix_contra_backward",
            "concat_greedy_semantic_matrix_contra",
        ]
        dependencies = ["greedy_texts", "no_fewshot_input_texts", "sample_texts"]
        return stats, dependencies

    def __init__(self, nli_model):
        super().__init__(nli_model)

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch_texts = dependencies["sample_texts"]
        batch_greedy_texts = dependencies["greedy_texts"]
        input_texts = dependencies["no_fewshot_input_texts"]

        batch_pairs = []
        batch_invs = []
        for texts, greedy_text, input_text in zip(
            batch_texts, batch_greedy_texts, input_texts
        ):
            texts = [input_text + text for text in texts]
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            batch_pairs.append(
                list(itertools.product([input_text + greedy_text], unique_texts))
            )
            batch_invs.append(inv)

        E_f, E_b, E, N_f, N_b, N, C_f, C_b, C = self.calculate_semantic_matrix(
            batch_pairs, batch_invs
        )

        return {
            "concat_greedy_semantic_matrix_entail_forward": E_f,
            "concat_greedy_semantic_matrix_entail_backward": E_b,
            "concat_greedy_semantic_matrix_entail": E,
            "concat_greedy_semantic_matrix_neutral_forward": N_f,
            "concat_greedy_semantic_matrix_neutral_backward": N_b,
            "concat_greedy_semantic_matrix_neutral": N,
            "concat_greedy_semantic_matrix_contra_forward": C_f,
            "concat_greedy_semantic_matrix_contra_backward": C_b,
            "concat_greedy_semantic_matrix_contra": C,
        }
