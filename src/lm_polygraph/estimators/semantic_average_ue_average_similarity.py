import numpy as np

from typing import Dict
from copy import deepcopy

from .estimator import Estimator


class SemanticAveMaxprobAveSimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "SemanticAveMaxprobAveSimilarityexp"
        else:
            return "SemanticAveMaxprobAveSimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        enriched_metrics = []  # To store enriched metrics for each sample

        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            sample_probs = -np.array(sample_log_probs)
            if self.exp:
                sample_probs = -np.exp(-sample_probs)

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_similarity = (np.sum(row) - row[i]) / (len(row) - 1)
                row_averages.append(average_similarity)

            # Enrich each metric by scaling it by 1/row_average
            enriched_sample_metrics = []
            for i, (prob, avg_similarity) in enumerate(zip(sample_probs, row_averages)):
                if avg_similarity == 0:
                    avg_similarity = 1e-10  # Avoid division by zero
                enriched_metric = prob * (1 / avg_similarity)
                enriched_sample_metrics.append(enriched_metric)

            enriched_metrics.append(np.array(enriched_sample_metrics))
        # Return only metric for the first sample for prr calculation
        first_elements = [metrics[0] for metrics in enriched_metrics]
        return np.array(first_elements)

class SemanticAvePPLAveSimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False
    ):
        super().__init__(["sample_sentence_similarity", "sample_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.exp = exp

    def __str__(self):
        if self.exp:
            return "SemanticAvePPLAveSimilarityexp"
        else:
            return "SemanticAvePPLAveSimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for sample_log_likelihoods, sample_sentence_similarity in zip(
            batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.array([np.mean(token_ll) for token_ll in sample_log_likelihoods])
            if self.exp:
                ppl = -np.exp(-ppl)

            #  Compute row-wise average similarity, excluding self-similarity
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_similarity = (np.sum(row) - row[i]) / (len(row) - 1)  # Exclude g_ii
                row_averages.append(average_similarity)

            # Enrich each PPL independently by scaling with 1/row_average
            enriched_sample_ppl = []
            for i, (ppl_value, avg_similarity) in enumerate(zip(ppl, row_averages)):
                if avg_similarity == 0:
                    avg_similarity = 1e-10  # Avoid division by zero
                enriched_value = ppl_value * (1 / avg_similarity)
                enriched_sample_ppl.append(enriched_value)

            enriched_ppl.append(np.array(enriched_sample_ppl))  # Collect enriched PPL values
        # Return only metric for the first sample for prr calculation
        first_elements = [metrics[0] for metrics in enriched_ppl]
        return np.array(first_elements)

class SemanticAveTokenSARAveSimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False
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

    def __str__(self):
        if self.exp:
            return "SemanticAveTokenSARAveSimilarityexp"
        else:
            return "SemanticAveTokenSARAveSimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        enriched_tokenSAR = []

        for batch_data in zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            batch_sample_sentence_similarity,
        ):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            sample_sentence_similarity = batch_data[2]

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

            # Compute row-wise average similarity excluding self-similarity
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_similarity = (np.sum(row) - row[i]) / (len(row) - 1)  # Exclude g_ii
                row_averages.append(average_similarity)

            # Enrich each tokenSAR value
            enriched_sample_tokenSAR = []
            for i, (sar_value, avg_similarity) in enumerate(zip(tokenSAR, row_averages)):
                if avg_similarity == 0:
                    avg_similarity = 1e-10  # Avoid division by zero
                enriched_value = sar_value * (1 / avg_similarity)
                enriched_sample_tokenSAR.append(enriched_value)

            enriched_tokenSAR.append(np.array(enriched_sample_tokenSAR))
        # Return only metric for the first sample for prr calculation

        first_elements = [metrics[0] for metrics in enriched_tokenSAR]
        return np.array(first_elements)


class SemanticAveMTEAveSimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
    ):
        super().__init__(["sample_sentence_similarity", "sample_entropy"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "SemanticAveMTEAveSimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        enriched_entropy = []

        for sample_entropy, sample_sentence_similarity in zip(
            batch_sample_entropy, batch_sample_sentence_similarity
        ):
            # Compute row-wise average similarity, excluding self-similarity
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_similarity = (np.sum(row) - row[i]) / (len(row) - 1)  # Exclude g_ii
                row_averages.append(average_similarity)

            # Enrich each sample's entropy value
            enriched_sample_entropy = []
            for i, (entropy, avg_similarity) in enumerate(zip(sample_entropy, row_averages)):
                if avg_similarity == 0:
                    avg_similarity = 1e-10  # Avoid division by zero
                enriched_value = entropy * (1 / avg_similarity)
                enriched_sample_entropy.append(enriched_value)

            enriched_entropy.append(np.array(enriched_sample_entropy))
        # Return only metric for the first sample for prr calculation
        first_elements = [metrics[0] for metrics in enriched_entropy]
        return np.array(first_elements)


