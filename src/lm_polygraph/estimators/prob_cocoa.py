import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS


class ProbCocoaMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_probs"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        base = "ProbCocoaMaxprob"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_metrics = []  # To store enriched metrics for each sample

        for best_id, sample_log_probs, sample_sentence_similarity in zip(
            sample_ids, batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sim = 1 - sample_sentence_similarity[best_id, :]
            sim[best_id] = 1
            avg_similarity = np.mean(sim)
            mp = 1 - np.exp(np.sum(sample_log_probs[best_id]))
            res = mp * avg_similarity
            enriched_metrics.append(res)

        return np.array(enriched_metrics)


class ProbCocoaPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_likelihoods"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        base = "ProbCocoaPPL"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_ppl = []  # To store enriched PPL for each sample

        for best_id, sample_log_likelihoods, sample_sentence_similarity in zip(
            sample_ids, batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            sim = 1 - sample_sentence_similarity[best_id, :]
            sim[best_id] = 1
            avg_similarity = np.mean(sim)
            ppl = 1 - np.exp(np.mean(sample_log_likelihoods[best_id]))
            res = ppl * avg_similarity
            enriched_ppl.append(res)

        return np.array(enriched_ppl)


class GreedyProbCocoaMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedyProbCocoaMaxprob"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = 1 - np.exp(greedy_ll)

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_similarity = 1 - np.mean(greedy_sentence_similarity)

            enriched_metrics.append(prob * avg_similarity)

        return np.array(enriched_metrics)


class GreedyProbCocoaPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedyProbCocoaPPL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = 1 - np.exp(np.mean(greedy_log_likelihoods))

            #  Compute row-wise average similarity, excluding self-similarity
            avg_similarity = 1 - np.mean(greedy_sentence_similarity)

            enriched_ppl.append(ppl * avg_similarity)


        return np.array(enriched_ppl)
