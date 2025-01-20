import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS


class SumSemanticMaxprob(Estimator):
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
        base = "SumSemanticMaxprob"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_metrics = []  # To store enriched metrics for each sample

        for best_id, sample_log_probs, sample_sentence_similarity in zip(
            sample_ids, batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sim = sample_sentence_similarity[best_id, :]
            sim[best_id] = 1
            avg_similarity = np.mean(sim)
            res = -np.sum(sample_log_probs[best_id]) - np.log(avg_similarity)
            enriched_metrics.append(res)

        return np.array(enriched_metrics)


class SumSemanticPPL(Estimator):
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
        base = "SumSemanticPPL"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_ppl = []  # To store enriched PPL for each sample

        for best_id, sample_log_likelihoods, sample_sentence_similarity in zip(
            sample_ids, batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            sim = sample_sentence_similarity[best_id, :]
            sim[best_id] = 1
            avg_similarity = np.mean(sim)
            res = -np.mean(sample_log_likelihoods[best_id]) - np.log(avg_similarity)
            enriched_ppl.append(res)

        return np.array(enriched_ppl)
