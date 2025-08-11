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


class StepsGreedyNLISimilarityCalculator(StatCalculator):
    """
    Calculates the NLI semantic matrix for generation samples using DeBERTa model.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["steps_greedy_nli_similarity"], [
            "sample_steps_texts",
            "claims",
        ]

    def __init__(self, nli_model):
        super().__init__()
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
        batch_texts: list[list[str]] = flatten(
            sample_steps_texts
        )  # batch_texts[step_idx][alternative_idx]
        greedy_texts: list[str] = [
            x.claim_text for x in flatten(dependencies["claims"])
        ]
        greedy_solutions: list[str] = [
            dependencies["greedy_texts"][i]
            for i in range(len(sample_steps_texts))
            for _ in sample_steps_texts[i]
        ]
        input_texts: list[str] = [
            texts[i]
            for i in range(len(sample_steps_texts))
            for _ in sample_steps_texts[i]
        ]
        assert len(batch_texts) == len(greedy_texts) == len(input_texts)

        batch_pairs = []
        for sample_texts, greedy_text, greedy_solution, input_text in zip(
            batch_texts, greedy_texts, greedy_solutions, input_texts
        ):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            greedy_step = self.parse_steps(greedy_text)
            sample_steps = [self.parse_steps(x) for x in sample_texts]
            prefix = "Problem: {problem}\nSolution: {solution} ".format(
                problem=self.parse_problem(input_text),
                solution=self.parse_solution(greedy_solution.split(greedy_text, 1)[0]),
            )
            batch_pairs += list(itertools.product([prefix + greedy_step], sample_steps))
            batch_pairs += list(
                itertools.product([prefix + x for x in sample_steps], [greedy_step])
            )
        original_order = batch_pairs.copy()
        batch_pairs = list(set(batch_pairs))

        device = deberta.device
        labels_dict = {
            "entailment": deberta.deberta.config.label2id["ENTAILMENT"],
            "contradiction": deberta.deberta.config.label2id["CONTRADICTION"],
            "neutral": deberta.deberta.config.label2id["NEUTRAL"],
        }

        softmax = nn.Softmax(dim=1)
        tokenizer = deberta.deberta_tokenizer

        dl = torch.utils.data.DataLoader(batch_pairs, batch_size=deberta_batch_size)
        probs = []
        for first_texts, second_texts in dl:
            batch = list(zip(first_texts, second_texts))
            encoded = tokenizer.batch_encode_plus(
                batch, padding=True, return_tensors="pt"
            ).to(device)
            logits = deberta.deberta(**encoded).logits.detach().to(device)
            probs.append(softmax(logits).cpu().detach())

        probs = torch.cat(probs, dim=0)
        prob_dict = {}
        for (first_text, second_text), prob in zip(batch_pairs, probs):
            prob_dict[first_text, second_text] = prob

        original_probs: list[dict] = [
            prob_dict[first_text, second_text]
            for first_text, second_text in original_order
        ]

        steps_greedy_nli_similarity = []
        for sample_texts, greedy_text in zip(batch_texts, greedy_texts):
            # Sampling from LLM often produces significant number of identical
            # outputs. We only need to score pairs of unqiue outputs
            steps_greedy_nli_similarity.append({})
            fwd_sample_probs = original_probs[: len(sample_texts)]
            original_probs = original_probs[len(sample_texts) :]
            bwd_sample_probs = original_probs[: len(sample_texts)]
            original_probs = original_probs[len(sample_texts) :]
            for k, k_id in labels_dict.items():
                steps_greedy_nli_similarity[-1][f"forward_{k}"] = [
                    f[k_id].item() for f in fwd_sample_probs
                ]
                steps_greedy_nli_similarity[-1][f"backward_{k}"] = [
                    f[k_id].item() for f in bwd_sample_probs
                ]
        return {
            "steps_greedy_nli_similarity": reconstruct(
                steps_greedy_nli_similarity, sample_steps_texts
            )
        }
