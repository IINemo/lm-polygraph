import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS


class AdjustedSumSemanticMaxprob(Estimator):
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
        base = "AdjustedSumSemanticMaxprob"
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
            mp = -np.sum(sample_log_probs[best_id])
            res = mp + avg_similarity * mp
            enriched_metrics.append(res)

        return np.array(enriched_metrics)


class AdjustedSumSemanticPPL(Estimator):
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
        base = "AdjustedSumSemanticPPL"
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
            ppl = -np.mean(sample_log_likelihoods[best_id])
            res = ppl + avg_similarity * ppl
            enriched_ppl.append(res)

        return np.array(enriched_ppl)


class AdjustedSumSemanticMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_entropy"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        base = "AdjustedSumSemanticMTE"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_entropies = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_mte = []  

        for best_id, sample_entropies, sample_sentence_similarity in zip(
            sample_ids, batch_entropies, batch_sample_sentence_similarity
        ):
            sim = 1 - sample_sentence_similarity[best_id, :]
            sim[best_id] = 1
            avg_similarity = np.mean(sim)
            mte = sample_entropies[best_id]
            res = mte + avg_similarity * mte
            enriched_mte.append(res)

        return np.array(enriched_mte)


class GreedyAdjustedSumSemanticMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedyAdjustedSumSemanticMaxprob"

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
            avg_similarity = 1 - np.mean(greedy_sentence_similarity)

            enriched_metrics.append(prob + avg_similarity * prob)

        return np.array(enriched_metrics)


class GreedyAdjustedSumSemanticPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedyAdjustedSumSemanticPPL"

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
            avg_similarity = 1 - np.mean(greedy_sentence_similarity)

            enriched_ppl.append(ppl + avg_similarity * ppl)


        return np.array(enriched_ppl)


class GreedyAdjustedSumSemanticMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedyAdjustedSumSemanticMTE"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropies = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_mte = []  # To store enriched PPL for each sample

        for greedy_entropies, greedy_sentence_similarity in zip(
            batch_greedy_entropies, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            mte = np.mean(greedy_entropies)

            #  Compute row-wise average similarity, excluding self-similarity
            avg_similarity = 1 - np.mean(greedy_sentence_similarity)

            enriched_mte.append(mte + avg_similarity * mte)


        return np.array(enriched_mte)
