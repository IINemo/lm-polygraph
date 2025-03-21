import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class GreedySemanticEnrichedMaxprobAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "GreedySemanticEnrichedMaxprobAveDissimilarityexp"
        else:
            return "GreedySemanticEnrichedMaxprobAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll
            if self.exp:
                prob = -np.exp(-prob)

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class SupervisedGreedySemanticEnrichedMaxprobAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "SupervisedGreedySemanticEnrichedMaxprobAveDissimilarityexp"
        else:
            return "SupervisedGreedySemanticEnrichedMaxprobAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity_pred"]
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll
            if self.exp:
                prob = -np.exp(-prob)

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = 1 - greedy_sentence_similarity

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class GreedySemanticEnrichedMaxprobTotalDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "GreedySemanticEnrichedMaxprobTotalDissimilarityexp"
        else:
            return "GreedySemanticEnrichedMaxprobTotalDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        batch_lls = np.array([np.sum(log_likelihood) for log_likelihood in stats["greedy_log_likelihoods"]])

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, sample_sentence_similarity in zip(
            batch_lls, batch_sample_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll
            if self.exp:
                prob = -np.exp(-prob)

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = np.mean(1 - np.array(sample_sentence_similarity))

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class GreedySemanticEnrichedPPLAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "GreedySemanticEnrichedPPLAveDissimilarityexp"
        else:
            return "GreedySemanticEnrichedPPLAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)
            if self.exp:
                ppl = -np.exp(-ppl)

            # Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_value = ppl *  avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class SupervisedGreedySemanticEnrichedPPLAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "SupervisedGreedySemanticEnrichedPPLAveDissimilarityexp"
        else:
            return "SupervisedGreedySemanticEnrichedPPLAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity_pred"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)
            if self.exp:
                ppl = -np.exp(-ppl)

            # Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = 1 - greedy_sentence_similarity

            enriched_value = ppl *  avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class GreedySemanticEnrichedPPLTotalDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "GreedySemanticEnrichedPPLTotalDissimilarityexp"
        else:
            return "GreedySemanticEnrichedPPLTotalDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, sample_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_sample_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)
            if self.exp:
                ppl = -np.exp(-ppl)

            # Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - np.array(sample_sentence_similarity))

            enriched_value = ppl *  avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class GreedySemanticEnrichedMTEAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedySemanticEnrichedMTEAveDissimilarity"

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


class SupervisedGreedySemanticEnrichedMTEAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "SupervisedGreedySemanticEnrichedMTEAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity_pred"]

        enriched_entropy = []

        for greedy_entropy, greedy_sentence_similarity in zip(
            batch_greedy_entropy, batch_greedy_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = 1 - greedy_sentence_similarity

            entropy = np.mean(greedy_entropy)
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)


class GreedySemanticEnrichedMTETotalDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedySemanticEnrichedMTETotalDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        batch_entropy = stats["entropy"]

        enriched_entropy = []

        for entropy, sample_sentence_similarity in zip(
            batch_entropy, batch_sample_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - np.array(sample_sentence_similarity))

            entropy = np.mean(entropy)
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)


class GreedyAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedyAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        res = []

        for greedy_entropy, greedy_sentence_similarity in zip(
            batch_greedy_entropy, batch_greedy_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)
            res.append(avg_dissimilarity)

        return np.array(res)
