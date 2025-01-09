import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class GreedySemanticAveMaxprobAveSimilarity(Estimator):
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
            return "GreedySemanticAveMaxprobAveSimilarityexp"
        else:
            return "GreedySemanticAveMaxprobAveSimilarity"

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
            ave_similarity = np.mean(greedy_sentence_similarity)

            # Enrich each metric by scaling it by 1/row_average
            if ave_similarity == 0:
                ave_similarity = 1e-10  # Avoid division by zero

            enriched_metric = prob * (1 / avg_similarity)
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


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
            return "GreedySemanticAveMaxprobAveDissimilarityexp"
        else:
            return "GreedySemanticAveMaxprobAveDissimilarity"

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
            ave_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class GreedySemanticAvePPLAveSimilarity(Estimator):
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
            return "GreedySemanticAvePPLAveSimilarityexp"
        else:
            return "GreedySemanticAvePPLAveSimilarity"

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

            #  Compute row-wise average similarity, excluding self-similarity
            avg_similarity = np.mean(greedy_sentence_similarity)

            # Enrich each PPL independently by scaling with 1/row_average
            if avg_similarity == 0:
                avg_similarity = 1e-10  # Avoid division by zero

            enriched_value = ppl * (1 / avg_similarity)
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


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
            return "GreedySemanticAvePPLAveDissimilarityexp"
        else:
            return "GreedySemanticAvePPLAveDissimilarity"

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


class GreedySemanticAveTokenSARAveSimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(
            [
                "greedy_sentence_similarity",
                "greedy_log_likelihoods",
            ],
            "sequence",
        )
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "GreedySemanticAveTokenSARAveSimilarityexp"
        else:
            return "GreedySemanticAveTokenSARAveSimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_token_similarity = stats["token_similarity"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_tokenSAR = []

        for batch_data in zip(
            batch_greedy_log_likelihoods,
            batch_greedy_token_similarity,
            batch_greedy_sentence_similarity,
        ):
            log_likelihoods = batch_data[0]
            token_similarity = batch_data[1]
            greedy_sentence_similarity = batch_data[2]

            log_likelihoods = np.array(log_likelihoods)
            R_t = 1 - token_similarity
            R_t_norm = R_t / R_t.sum()
            E_t = -log_likelihoods * R_t_norm
            tokenSAR.append(E_t.sum())

            if self.exp:
                tokenSAR = -np.exp(-np.array(tokenSAR))

            #  Compute row-wise average similarity, excluding self-similarity
            avg_similarity = np.mean(greedy_sentence_similarity)

            # Enrich each PPL independently by scaling with 1/row_average
            if avg_similarity == 0:
                avg_similarity = 1e-10  # Avoid division by zero

            enriched_value = tokenSAR * (1 / avg_similarity)
            enriched_tokenSAR.append(enriched_value)

        return np.array(enriched_tokenSAR)


class GreedySemanticEnrichedTokenSARAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
    ):
        super().__init__(
            [
                "greedy_sentence_similarity",
                "greedy_log_likelihoods",
            ],
            "sequence",
        )
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "GreedySemanticAveTokenSARAveDissimilarityexp"
        else:
            return "GreedySemanticAveTokenSARAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_token_similarity = stats["token_similarity"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_tokenSAR = []

        for batch_data in zip(
            batch_greedy_log_likelihoods,
            batch_greedy_token_similarity,
            batch_greedy_sentence_similarity,
        ):
            log_likelihoods = batch_data[0]
            token_similarity = batch_data[1]
            greedy_sentence_similarity = batch_data[2]

            log_likelihoods = np.array(log_likelihoods)
            R_t = 1 - token_similarity
            R_t_norm = R_t / R_t.sum()
            E_t = -log_likelihoods * R_t_norm
            tokenSAR.append(E_t.sum())

            if self.exp:
                tokenSAR = -np.exp(-np.array(tokenSAR))

            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - greedy_sentence_similarity)

            enriched_value = tokenSAR * avg_dissimilarity
            enriched_tokenSAR.append(enriched_value)

        return np.array(enriched_tokenSAR)


class GreedySemanticAveMTEAveSimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["greedy_sentence_similarity", "entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "GreedySemanticAveMTEAveSimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_greedy_sentence_similarity = stats["greedy_sentence_similarity"]

        enriched_entropy = []

        for greedy_entropy, greedy_sentence_similarity in zip(
            batch_greedy_entropy, batch_greedy_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_similarity = np.mean(greedy_sentence_similarity)

            # Enrich each PPL independently by scaling with 1/row_average
            if avg_similarity == 0:
                avg_similarity = 1e-10  # Avoid division by zero

            enriched_value = greedy_entropy * (1 / avg_similarity)
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)


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

            enriched_value = greedy_entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)
