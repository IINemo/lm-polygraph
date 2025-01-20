import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class GreedySumSemanticMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedySumSemanticMaxprob"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_similarity = np.mean(greedy_sentence_similarity)

            enriched_metrics.append(prob - np.log(avg_similarity))

        return np.array(enriched_metrics)


class GreedySumSemanticPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedySumSemanticPPL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)

            #  Compute row-wise average similarity, excluding self-similarity
            avg_similarity = np.mean(greedy_sentence_similarity)

            enriched_ppl.append(ppl - np.log(avg_similarity))


        return np.array(enriched_ppl)
