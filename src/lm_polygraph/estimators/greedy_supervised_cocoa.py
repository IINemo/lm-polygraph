import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class SupervisedCocoaMSP(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity_supervised", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "SupervisedCocoaMSPexp"
        else:
            return "SupervisedCocoaMSP"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_avg_dissimilarity = stats["greedy_sentence_similarity_supervised"]
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])

        enriched_metrics = []
        for greedy_ll, avg_dissimilarity in zip(batch_lls, batch_avg_dissimilarity):
            prob = -greedy_ll
            if self.exp:
                prob = -np.exp(-prob)

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class SupervisedCocoaPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity_supervised", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "SupervisedCocoaPPLexp"
        else:
            return "SupervisedCocoaPPL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_avg_dissimilarity = stats["greedy_sentence_similarity_supervised"]

        enriched_ppl = []

        for greedy_log_likelihoods, avg_dissimilarity in zip(
            batch_greedy_log_likelihoods, batch_avg_dissimilarity
        ):
            ppl = -np.mean(greedy_log_likelihoods)
            if self.exp:
                ppl = -np.exp(-ppl)

            enriched_value = ppl * avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class SupervisedCocoaMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity_supervised", "entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "SupervisedCocoaMTE"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_avg_dissimilarity = stats["greedy_sentence_similarity_supervised"]

        enriched_entropy = []

        for greedy_entropy, avg_dissimilarity in zip(
            batch_greedy_entropy, batch_avg_dissimilarity
        ):
            entropy = np.mean(greedy_entropy)
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)



class SupervisedCocoa(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity_supervised"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "SupervisedCocoa"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array(stats["greedy_sentence_similarity_supervised"])

