import numpy as np
import re

import itertools
from typing import Dict, List, Tuple, Union

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
import torch.nn as nn
import torch

from .utils import flatten, reconstruct

softmax = nn.Softmax(dim=1)


class StepsSemanticMatrixCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "steps_semantic_matrix_entail",
            "steps_semantic_matrix_contra",
            "steps_semantic_matrix_classes",
            "steps_semantic_matrix_entail_logits",
            "steps_semantic_matrix_contra_logits",
            "entailment_id",
        ], ["sample_steps_texts"]

    def __init__(self, nli_model):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model

    def parse_steps(self, x: str) -> str:
        x = re.sub(r"- Step \d+:\s*", "", x)
        x = x.replace("<Answer>:", "Answer:")
        return x

    def parse_problem(self, x: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(x, str):
            return (
                x.split("<Question>: ", 1)[-1]
                .split("<|im_end|>", 1)[0]
                .replace("  ", " ")
                .strip()
            )
        else:
            return (
                x[0]["content"]
                .split("<Question>: ", 1)[-1]
                .split("<|im_end|>", 1)[0]
                .replace("  ", " ")
                .strip()
            )

    def parse_solution(self, x: str) -> str:
        x = x.split("Reasoning Steps:\n")[-1].strip().replace("\n", " ")
        return self.parse_steps(x)

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
        sample_steps_texts = dependencies["sample_steps_texts"]
        batch_texts: list[list[str]] = flatten(sample_steps_texts)
        greedy_texts: list[str] = [
            x.claim_text for x in flatten(dependencies["claims"])
        ]
        input_texts: list[str] = [
            texts[i]
            for i in range(len(sample_steps_texts))
            for _ in sample_steps_texts[i]
        ]
        greedy_solutions: list[str] = [
            dependencies["greedy_texts"][i]
            for i in range(len(sample_steps_texts))
            for _ in sample_steps_texts[i]
        ]

        batch_pairs = []
        batch_invs = []
        batch_counts = []
        assert (
            len(input_texts)
            == len(greedy_solutions)
            == len(greedy_texts)
            == len(batch_texts)
        )
        for input_text, greedy_solution, greedy_step, texts in zip(
            input_texts, greedy_solutions, greedy_texts, batch_texts
        ):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            sample_steps = [self.parse_steps(s) for s in texts]
            prefix = "Problem: {problem}\nSolution: {solution} ".format(
                problem=self.parse_problem(input_text),
                solution=self.parse_solution(greedy_solution.split(greedy_step, 1)[0]),
            )
            unique_texts, inv = np.unique(sample_steps, return_inverse=True)
            batch_pairs.append(
                list(
                    itertools.product([prefix + x for x in unique_texts], unique_texts)
                )
            )
            batch_invs.append(inv)
            batch_counts.append(len(unique_texts))

        device = deberta.device
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        E = []
        C = []
        E_logits = []
        C_logits = []
        P = []

        for i, pairs in enumerate(batch_pairs):
            dl = torch.utils.data.DataLoader(pairs, batch_size=deberta_batch_size)
            probs = []
            logits_all = []
            for first_texts, second_texts in dl:
                batch = list(zip(first_texts, second_texts))
                encoded = tokenizer.batch_encode_plus(
                    batch, padding=True, return_tensors="pt"
                ).to(device)
                logits = deberta.deberta(**encoded).logits.detach().to(device)
                probs.append(softmax(logits).cpu().detach())
                logits_all.append(logits.cpu().detach())
            probs = torch.cat(probs, dim=0)
            logits_all = torch.cat(logits_all, dim=0)

            entail_probs = probs[:, ent_id]
            contra_probs = probs[:, contra_id]
            entail_logits = logits_all[:, ent_id]
            contra_logits = logits_all[:, contra_id]
            class_preds = probs.argmax(-1)

            unique_mat_shape = (batch_counts[i], batch_counts[i])

            unique_E = entail_probs.view(unique_mat_shape).numpy()
            unique_C = contra_probs.view(unique_mat_shape).numpy()
            unique_E_logits = entail_logits.view(unique_mat_shape).numpy()
            unique_C_logits = contra_logits.view(unique_mat_shape).numpy()
            unique_P = class_preds.view(unique_mat_shape).numpy()

            inv = batch_invs[i]

            # Recover full matrices from unques by gathering along both axes
            # using inverse index
            E.append(unique_E[inv, :][:, inv])
            C.append(unique_C[inv, :][:, inv])
            E_logits.append(unique_E_logits[inv, :][:, inv])
            C_logits.append(unique_C_logits[inv, :][:, inv])
            P.append(unique_P[inv, :][:, inv])

        return {
            "steps_semantic_matrix_entail": reconstruct(E, sample_steps_texts),
            "steps_semantic_matrix_contra": reconstruct(C, sample_steps_texts),
            "steps_semantic_matrix_entail_logits": reconstruct(
                E_logits, sample_steps_texts
            ),
            "steps_semantic_matrix_contra_logits": reconstruct(
                C_logits, sample_steps_texts
            ),
            "steps_semantic_matrix_classes": reconstruct(P, sample_steps_texts),
            "entailment_id": deberta.deberta.config.label2id["ENTAILMENT"],
        }
