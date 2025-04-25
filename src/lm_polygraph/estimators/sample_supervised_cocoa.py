import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class SampledSupervisedCocoaMSP(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["supervised_sample_sentence_similarity", "sample_log_probs"],
            "sequence"
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        base = "SampledSupervisedCocoaPPLexp" if self.exp else "SampledSupervisedCocoaPPL"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs_batch = stats["sample_log_probs"]
        sim_scores = stats["supervised_sample_sentence_similarity"]  # Shape: (batch_size,)
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        results = []

        for log_probs, sim, best_id in zip(log_probs_batch, sim_scores, sample_ids):
            prob = -log_probs[best_id]
            if self.exp:
                prob = -np.exp(-prob)

            dissim = 1 - sim  # already corresponds to best_id
            enriched_metric = prob * dissim
            results.append(enriched_metric)

        return np.array(results)


class SampledSupervisedCocoaPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["supervised_sample_sentence_similarity", "sample_log_likelihoods"],
            "sequence"
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        base = "SemanticEnrichedPPLAveDissimilarityexp" if self.exp else "SemanticEnrichedPPLAveDissimilarity"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        sim_scores = stats["supervised_sample_sentence_similarity"]  # shape: (batch,)
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        results = []

        for token_lls, sim, best_id in zip(batch_sample_log_likelihoods, sim_scores, sample_ids):
            ppl = -np.mean(token_lls[best_id])
            if self.exp:
                ppl = -np.exp(-ppl)

            dissim = 1 - sim
            enriched = ppl * dissim

            results.append(enriched)

        return np.array(results)

class SampledSupervisedCocoaMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["supervised_sample_sentence_similarity", "sample_entropy"] ,
            "sequence"
        )
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        return sample_strategy_to_prefix(self.sample_strategy) + "SemanticEnrichedMTEAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]  # shape: (batch_size, num_samples)
        sim_scores = stats["supervised_sample_sentence_similarity"]  # shape: (batch_size,)
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        results = []

        for entropy_list, sim, best_id in zip(batch_sample_entropy, sim_scores, sample_ids):
            entropy = entropy_list[best_id]
            dissim = 1 - sim
            enriched_value = entropy * dissim
            results.append(enriched_value)

        return np.array(results)



class SampledSupervisedCocoa(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["supervised_sample_sentence_similarity"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "SampledSupervisedCocoa"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array(stats["supervised_sample_sentence_similarity"])

