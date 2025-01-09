import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids

from wquantiles import median


class SemanticMedianMaxprob(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticMedianMaxprobexp"
        else:
            base = "SemanticMedianMaxprob"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        ave = []
        for best_id, sample_log_probs, sample_sentence_similarity in zip(
            sample_ids, batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sample_probs = -np.array(sample_log_probs)
            if self.exp:
                sample_probs = -np.exp(-sample_probs)
            weights = sample_sentence_similarity[best_id, :]
            ave.append(median(sample_probs, weights))

        return np.array(ave)

class SemanticMedianPPL(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticMedianPPLexp"
        else:
            base = "SemanticMedianPPL"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        ave = []
        for best_id, sample_log_likelihoods, sample_sentence_similarity in zip(
            sample_ids, batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            ppl = -np.array([np.mean(token_ll) for token_ll in sample_log_likelihoods])

            if self.exp:
                ppl = -np.exp(-ppl)

            weights = sample_sentence_similarity[best_id, :]

            ave.append(median(ppl, weights))

        return np.array(ave)

class SemanticMedianTokenSAR(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            [
                "sample_sentence_similarity",
                "sample_log_likelihoods",
                "sample_token_similarity",
            ],
            "sequence",
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticMedianTokenSARexp"
        else:
            base = "SemanticMedianTokenSAR"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        ave = []
        for batch_data in zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            batch_sample_sentence_similarity,
            sample_ids,
        ):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            sample_sentence_similarity = batch_data[2]
            best_id = batch_data[3]

            tokenSAR = []
            for log_likelihoods, token_similarity in zip(
                sample_log_likelihoods, sample_token_similarity
            ):
                log_likelihoods = np.array(log_likelihoods)
                R_t = 1 - token_similarity
                R_t_norm = R_t / R_t.sum()
                E_t = -log_likelihoods * R_t_norm
                tokenSAR.append(E_t.sum())

            if self.exp:
                tokenSAR = -np.exp(-np.array(tokenSAR))

            weights = sample_sentence_similarity[best_id, :]

            ave.append(median(np.array(tokenSAR), weights))

        return np.array(ave)

class SemanticMedianMTE(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(["sample_sentence_similarity", "sample_entropy"], "sequence")
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        return sample_strategy_to_prefix(self.sample_strategy) + "SemanticMedianMTE"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        ave = []
        for best_id, sample_entropy, sample_sentence_similarity in zip(
            sample_ids, batch_sample_entropy, batch_sample_sentence_similarity
        ):
            weights = sample_sentence_similarity[best_id, :]
            ave.append(median(np.array(sample_entropy), weights))

        return np.array(ave)
