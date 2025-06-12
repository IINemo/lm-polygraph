import numpy as np

from typing import Dict

from .estimator import Estimator


class CocoaMSP(Estimator):
    def __init__(
        self,
    ):
        super().__init__(
            ["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence"
        )

    def __str__(self):
        return "CocoaMSP"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]
        batch_lls = np.array(
            [
                np.sum(log_likelihood)
                for log_likelihood in stats["greedy_log_likelihoods"]
            ]
        )

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class CocoaPPL(Estimator):
    def __init__(
        self,
    ):
        super().__init__(
            ["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence"
        )

    def __str__(self):
        return "CocoaPPL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)

            # Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_value = ppl * avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class CocoaMTE(Estimator):
    def __init__(
        self,
    ):
        super().__init__(["greedy_sentence_similarity", "entropy"], "sequence")

    def __str__(self):
        return "CocoaMTE"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_entropy = []

        for greedy_entropy, greedy_sentence_similarity in zip(
            batch_greedy_entropy, batch_greedy_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            entropy = np.mean(greedy_entropy)
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)
