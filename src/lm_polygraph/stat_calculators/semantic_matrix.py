import numpy as np

import itertools
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)


class SemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "semantic_matrix_entail",
            "semantic_matrix_contra",
            "semantic_matrix_classes",
            "semantic_matrix_entail_logits",
            "semantic_matrix_contra_logits",
            "entailment_id",
        ], ["sample_texts"]

    def __init__(self, nli_model):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the NLI semantic matrix for generation samples using DeBERTa model.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, containing:
                - 'sample_texts' (List[List[str]]): several sampling generations
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
                - 'semantic_matrix_entail_logits' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with logits of 'ENTAILMENT' output of DeBERTa.
                - 'semantic_matrix_contra_logits' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with logits of 'CONTRADICTION' output of DeBERTa.
                - 'semantic_matrix_classes' (List[np.array]): for each input text: quadratic matrix of size
                    n_samples x n_samples, with the NLI label id corresponding to the DeBERTa prediction.
        """

        deberta = self.nli_model
        deberta_batch_size = deberta.batch_size
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

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E_tensors = []
        C_tensors = []
        E_logits_tensors = []
        C_logits_tensors = []
        P_tensors = []

        with torch.no_grad():
            for i, pairs in enumerate(batch_pairs):
                dl = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
                probs = []
                logits_all = []
                for first_texts, second_texts in dl:
                    batch = list(zip(first_texts, second_texts))
                    encoded = tokenizer.batch_encode_plus(
                        batch, padding=True, return_tensors="pt"
                    ).to(device)
                    logits = deberta.deberta(**encoded).logits
                    probs.append(softmax(logits))
                    logits_all.append(logits)
                probs = torch.cat(probs, dim=0)
                logits_all = torch.cat(logits_all, dim=0)

                del encoded, logits
                torch.cuda.empty_cache()

                entail_probs = probs[:, ent_id]
                contra_probs = probs[:, contra_id]
                entail_logits = logits_all[:, ent_id]
                contra_logits = logits_all[:, contra_id]
                class_preds = probs.argmax(-1)

                mat_shape = (batch_counts[i], batch_counts[i])
                unique_E = entail_probs.view(mat_shape)
                unique_C = contra_probs.view(mat_shape)
                unique_E_logits = entail_logits.view(mat_shape)
                unique_C_logits = contra_logits.view(mat_shape)
                unique_P = class_preds.view(mat_shape)

                inv = batch_invs[i]
                inv_tensor = torch.as_tensor(inv, device=device)

                # Recover full matrices by indexing with inverse indices
                E_tensors.append(unique_E[inv_tensor, :][:, inv_tensor])
                C_tensors.append(unique_C[inv_tensor, :][:, inv_tensor])
                E_logits_tensors.append(unique_E_logits[inv_tensor, :][:, inv_tensor])
                C_logits_tensors.append(unique_C_logits[inv_tensor, :][:, inv_tensor])
                P_tensors.append(unique_P[inv_tensor, :][:, inv_tensor])

            E = torch.stack(E_tensors)
            C = torch.stack(C_tensors)
            E_logits = torch.stack(E_logits_tensors)
            C_logits = torch.stack(C_logits_tensors)
            P = torch.stack(P_tensors)

        # Convert to numpy arrays on CPU at the end of computation
        E = E.cpu().numpy()
        C = C.cpu().numpy()
        E_logits = E_logits.cpu().numpy()
        C_logits = C_logits.cpu().numpy()
        P = P.cpu().numpy()

        return {
            "semantic_matrix_entail": E,
            "semantic_matrix_contra": C,
            "semantic_matrix_entail_logits": E_logits,
            "semantic_matrix_contra_logits": C_logits,
            "semantic_matrix_classes": P,
            "entailment_id": ent_id,
        }
