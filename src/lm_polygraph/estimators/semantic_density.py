import numpy as np

from typing import Dict

from .estimator import Estimator


class SemanticDensity(Estimator):

    def __init__(self, verbose: bool = False, concat_input: bool = True):
        deps = [
            "greedy_log_probs",
            "sample_log_probs",
            "sample_tokens",
            "sample_texts",
        ]
        if concat_input:
            deps.extend(
                [
                    "concat_greedy_semantic_matrix_contra_forward",
                    "concat_greedy_semantic_matrix_neutral_forward",
                ]
            )
        else:
            deps.extend(
                [
                    "greedy_semantic_matrix_contra_forward",
                    "greedy_semantic_matrix_neutral_forward",
                ]
            )
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.concat_input = concat_input

    def __str__(self):
        if self.concat_input:
            return "SemanticDensity"
        else:
            return "SemanticDensityOutput"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_tokens = stats["sample_tokens"]
        batch_sample_texts = stats["sample_texts"]
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]

        if self.concat_input:
            batch_semantic_matrix_contra = stats[
                "concat_greedy_semantic_matrix_contra_forward"
            ]
            batch_semantic_matrix_neutral = stats[
                "concat_greedy_semantic_matrix_neutral_forward"
            ]
        else:
            batch_semantic_matrix_contra = stats[
                "greedy_semantic_matrix_contra_forward"
            ]
            batch_semantic_matrix_neutral = stats[
                "greedy_semantic_matrix_neutral_forward"
            ]

        semantic_density = []
        for batch_data in zip(
            batch_greedy_log_likelihoods,
            batch_sample_log_probs,
            batch_sample_tokens,
            batch_sample_texts,
            batch_semantic_matrix_contra,
            batch_semantic_matrix_neutral,
        ):
            greedy_log_probs = batch_data[0]
            sample_probs = np.exp(batch_data[1])
            sample_tokens = batch_data[2]
            sample_texts = batch_data[3]
            semantic_matrix_contra = batch_data[4]
            semantic_matrix_neutral = batch_data[5]

            _, unique_sample_indices = np.unique(sample_texts, return_index=True)

            numerator, denominator = [], []

            for _id in unique_sample_indices:
                normed_prob = sample_probs[_id] ** (1 / len(sample_tokens[_id]))
                distance = semantic_matrix_contra[_id] + (
                    semantic_matrix_neutral[_id] / 2
                )

                if distance <= 1:
                    kernel_value = 1 - distance
                else:
                    kernel_value = 0

                numerator.append(normed_prob * kernel_value)
                denominator.append(normed_prob)

            greedy_normed_prob = np.exp(np.sum(greedy_log_probs)) ** (
                1 / len(greedy_log_probs)
            )
            numerator.append(greedy_normed_prob)
            denominator.append(greedy_normed_prob)

            semantic_density.append(np.sum(numerator) / np.sum(denominator))

        return -np.array(semantic_density)
