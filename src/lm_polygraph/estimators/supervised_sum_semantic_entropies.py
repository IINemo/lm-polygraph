import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS
from sklearn.preprocessing import MinMaxScaler


def get_avg_dissim(sample_sentence_similarity, sample_ids):
    batch_avg_similarity = []
    for best_id, sentence_similarity in zip(sample_ids, sample_sentence_similarity):
        batch_avg_similarity.append(np.mean(1 - sentence_similarity[best_id, :]))
    return batch_avg_similarity

def normalize_and_enrich(batch_metrics, batch_avg_dissimilarity, alpha):
    batch_metrics = MinMaxScaler().fit_transform(np.array(batch_metrics).reshape(-1, 1)).flatten()
    batch_avg_dissimilarity = MinMaxScaler().fit_transform(np.array(batch_avg_dissimilarity).reshape(-1, 1)).flatten()
    enriched_metrics = [metric + avg_dissimilarity * alpha for metric, avg_dissimilarity in zip(batch_metrics, batch_avg_dissimilarity)]
    return enriched_metrics


class SupSumSemanticMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first",
        alpha: int = 1
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_probs"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy
        self.alpha = alpha

    def __str__(self):
        base = f"SupSumSemanticMaxprob_{self.alpha}"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        batch_mps = [-np.sum(log_probs[best_id]) for best_id, log_probs in zip(sample_ids, stats["sample_log_probs"])]
        batch_avg_dissim = get_avg_dissim(stats["sample_sentence_similarity"], sample_ids)

        enriched_metrics = normalize_and_enrich(batch_mps, batch_avg_dissim, self.alpha)

        return np.array(enriched_metrics)


class SupSumSemanticPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first",
        alpha: int = 1 
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_likelihoods"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy
        self.alpha = alpha

    def __str__(self):
        base = f"SupSumSemanticPPL_{self.alpha}"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        batch_ppls = [-np.mean(sample_log_likelihoods[best_id]) for best_id, sample_log_likelihoods in zip(sample_ids, stats["sample_log_likelihoods"])]
        batch_avg_dissim = get_avg_dissim(stats["sample_sentence_similarity"], sample_ids)
        
        enriched_metrics = normalize_and_enrich(batch_ppls, batch_avg_dissim, self.alpha)

        return np.array(enriched_metrics)


class SupSumSemanticMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first",
        alpha: int = 1
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_entropy"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy
        self.alpha = alpha

    def __str__(self):
        base = f"SupSumSemanticMTE_{self.alpha}"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        batch_mtes = [entropies[best_id] for best_id, entropies in zip(sample_ids, stats["sample_entropy"])]
        batch_avg_dissim = get_avg_dissim(stats["sample_sentence_similarity"], sample_ids)

        enriched_metrics = normalize_and_enrich(batch_mtes, batch_avg_dissim, self.alpha)

        return np.array(enriched_metrics)


class GreedySupSumSemanticMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        alpha: int = 1
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.alpha = alpha

    def __str__(self):
        return f"GreedySupSumSemanticMaxprob_{self.alpha}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])
        batch_avg_dissim = [np.mean(1 - sentence_similarity) for sentence_similarity in stats["greedy_sentence_similarity"]]
        
        enriched_metrics = normalize_and_enrich(batch_lls, batch_avg_dissim, self.alpha)

        return np.array(enriched_metrics)


class GreedySupSumSemanticPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        alpha: int = 1
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.alpha = alpha

    def __str__(self):
        return f"GreedySupSumSemanticPPL_{self.alpha}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_ppls = [-np.mean(greedy_log_likelihoods) for greedy_log_likelihoods in stats["greedy_log_likelihoods"]]
        batch_avg_dissim = [np.mean(1 - sentence_similarity) for sentence_similarity in stats["greedy_sentence_similarity"]]

        enriched_metrics = normalize_and_enrich(batch_ppls, batch_avg_dissim, self.alpha)

        return np.array(enriched_metrics)


class GreedySupSumSemanticMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        alpha: int = 1
    ):
        super().__init__(["greedy_sentence_similarity", "entropy"], "sequence")
        self.verbose = verbose
        self.alpha = alpha

    def __str__(self):
        return f"GreedySupSumSemanticMTE_{self.alpha}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        #batch_greedy_entropies = stats["greedy_log_likelihoods"]
        #batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]
        batch_mtes = [np.mean(greedy_entropies) for greedy_entropies in stats["greedy_log_likelihoods"]]
        batch_avg_dissim = [np.mean(1 - sentence_similarity) for sentence_similarity in stats["greedy_sentence_similarity"]]

        enriched_metrics = normalize_and_enrich(batch_mtes, batch_avg_dissim, self.alpha)

        return np.array(enriched_metrics)
