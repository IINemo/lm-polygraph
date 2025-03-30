import numpy as np

from typing import Dict

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS


class SemanticDensity(Estimator):
    def __init__(self, verbose: bool = False, sample_strategy: str = "first"):
        super().__init__(
            [
                "sample_log_probs",
                "sample_tokens",
                "sample_texts",
                "concat_semantic_matrix_contra",
                "concat_semantic_matrix_neutral",
            ] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence",
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        base = "SemanticDensity"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_ids = best_sample_ids(self.sample_strategy, stats)
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_tokens = stats["sample_tokens"]
        batch_sample_texts = stats["sample_texts"]
        batch_semantic_matrix_contra = stats["concat_semantic_matrix_contra"]
        batch_semantic_matrix_neutral = stats["concat_semantic_matrix_neutral"]

        semantic_density = []
        for batch_data in zip(
            batch_sample_ids,
            batch_sample_log_probs,
            batch_sample_tokens,
            batch_sample_texts,
            batch_semantic_matrix_contra,
            batch_semantic_matrix_neutral,
        ):
            sample_id = batch_data[0]
            sample_probs = np.exp(batch_data[1])
            sample_tokens = batch_data[2]
            sample_texts = batch_data[3]
            semantic_matrix_contra = batch_data[4]
            semantic_matrix_neutral = batch_data[5]

            _, unique_sample_indices = np.unique(sample_texts, return_index=True)

            numerator, denominator = [], []

            for _id in unique_sample_indices:
                normed_prob = sample_probs[_id] ** (1 / len(sample_tokens[_id]))
                distance = semantic_matrix_contra[sample_id, _id] + (semantic_matrix_neutral[sample_id, _id] / 2)

                if distance <= 1:
                    kernel_value = 1 - distance
                else:
                    kernel_value = 0

                numerator.append(normed_prob * kernel_value)
                denominator.append(normed_prob)

            semantic_density.append(np.sum(numerator) / np.sum(denominator))

        return -np.array(semantic_density)


class GreedySemanticDensity(Estimator):
    def __init__(self, verbose: bool = False):
        super().__init__(
            [
                "greedy_log_probs",
                "sample_log_probs",
                "sample_tokens",
                "sample_texts",
                "concat_greedy_semantic_matrix_contra_forward",
                "concat_greedy_semantic_matrix_neutral_forward",
            ],
            "sequence",
        )
        self.verbose = verbose

    def __str__(self):
        return "GreedySemanticDensity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_tokens = stats["sample_tokens"]
        batch_sample_texts = stats["sample_texts"]
        batch_semantic_matrix_contra = stats["concat_greedy_semantic_matrix_contra_forward"]
        batch_semantic_matrix_neutral = stats["concat_greedy_semantic_matrix_neutral_forward"]
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]

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
                distance = semantic_matrix_contra[_id] + (semantic_matrix_neutral[_id] / 2)

                if distance <= 1:
                    kernel_value = 1 - distance
                else:
                    kernel_value = 0

                numerator.append(normed_prob * kernel_value)
                denominator.append(normed_prob)

            greedy_normed_prob = np.exp(np.sum(greedy_log_probs)) ** (1 / len(greedy_log_probs))
            numerator.append(greedy_normed_prob)
            denominator.append(greedy_normed_prob)

            semantic_density.append(np.sum(numerator) / np.sum(denominator))

        return -np.array(semantic_density)
