import numpy as np

import itertools
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)


class SemanticMatrixCalculator(StatCalculator):
    def _calculate_matrices(
        self,
        batch_pairs,
        batch_invs,
        batch_counts
    ):
        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E = []
        C = []
        P = []

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

            entail_probs = probs[:, ent_id]
            contra_probs = probs[:, contra_id]
            class_preds = probs.argmax(-1)

            unique_mat_shape = (batch_counts[i], batch_counts[i])

            unique_E = entail_probs.view(unique_mat_shape).numpy()
            unique_C = contra_probs.view(unique_mat_shape).numpy()
            unique_P = class_preds.view(unique_mat_shape).numpy()

            inv = batch_invs[i]

            # Recover full matrices from unques by gathering along both axes
            # using inverse index
            E.append(unique_E[inv, :][:, inv])
            C.append(unique_C[inv, :][:, inv])
            P.append(unique_P[inv, :][:, inv])

        E = np.stack(E)
        C = np.stack(C)
        P = np.stack(P)

        return E, C, P


    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the NLI semantic matrix for generation samples using DeBERTa model.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'blackbox_sample_texts' (List[List[str]]): several sampling generations
                    for each input text in the batch.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'semantic_matrix_entail' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with probabilities of 'ENTAILMENT' output of DeBERTa.
                - 'semantic_matrix_contra' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with probabilities of 'CONTRADICTION' output of DeBERTa.
                - 'semantic_matrix_classes' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with the NLI label id corresponding to the DeBERTa prediction.
        """

        batch_texts = dependencies["blackbox_sample_texts"]
        batch_inputs = dependencies["input_texts"]

        batch_pairs = []
        batch_invs = []
        batch_counts = []
        for input_text, texts in zip(batch_inputs, batch_texts):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            unique_texts, inv = np.unique(texts, return_inverse=True)
            if self.concat_input_output:
                unique_texts = [input_text + text for text in unique_texts]
            batch_pairs.append(list(itertools.product(unique_texts, unique_texts)))
            batch_invs.append(inv)
            batch_counts.append(len(unique_texts))

        E, C, P = self._calculate_matrices(
            batch_pairs,
            batch_invs,
            batch_counts
        )

        res = {}

        if self.concat_input_output:
            res["input_output_semantic_matrix_entail"] = E
            res["input_output_semantic_matrix_contra"] = C
            res["input_output_semantic_matrix_classes"] = P
        else:
            res["semantic_matrix_entail"] = E
            res["semantic_matrix_contra"] = C
            res["semantic_matrix_classes"] = P

        res["entailment_id"] = self.nli_model.deberta.config.label2id["ENTAILMENT"],

        return res


class InputOutputSemanticMatrixCalculator(SemanticMatrixCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model
    concatenated with input texts.
    """

    def __init__(self, nli_model):
        super().__init__(
            [
                "input_output_semantic_matrix_entail",
                "input_output_semantic_matrix_contra",
                "input_output_semantic_matrix_classes",
                "entailment_id",
            ],
            ["blackbox_sample_texts", "input_texts"],
        )
        self.concat_input_output = True
        self.is_deberta_setup = False
        self.nli_model = nli_model


class OutputSemanticMatrixCalculator(SemanticMatrixCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    def __init__(self, nli_model):
        super().__init__(
            [
                "semantic_matrix_entail",
                "semantic_matrix_contra",
                "semantic_matrix_classes",
                "entailment_id",
            ],
            ["blackbox_sample_texts", "input_texts"],
        )
        self.concat_input_output = False
        self.is_deberta_setup = False
        self.nli_model = nli_model
