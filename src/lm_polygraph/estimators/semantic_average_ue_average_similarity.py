import numpy as np

from typing import Dict

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS


class SemanticEnrichedMaxprobAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_probs"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticEnrichedMaxprobAveDissimilarityexp"
        else:
            base = "SemanticEnrichedMaxprobAveDissimilarity"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_metrics = []  # To store enriched metrics for each sample

        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            # Step 1: Compute probabilities (negative log-probs)
            sample_probs = -np.array(sample_log_probs)
            if self.exp:
                sample_probs = -np.exp(-sample_probs)

            # Step 2: Compute row-wise sum of dissimilarities (1 - g)
            row_dissimilarities = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                sum_dissimilarities = np.sum(1 - row) - (1 - row[i])  # Exclude self-similarity
                row_dissimilarities.append(sum_dissimilarities)

            # Step 3: Normalize by (M - 1)
            normalized_dissimilarities = [
                dissim / (len(sample_sentence_similarity) - 1)
                for dissim in row_dissimilarities
            ]

            # Step 4: Enrich each metric
            enriched_sample_metrics = []
            for prob, dissim in zip(sample_probs, normalized_dissimilarities):
                enriched_metric = prob * dissim
                enriched_sample_metrics.append(enriched_metric)

            enriched_metrics.append(np.array(enriched_sample_metrics))

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_metrics):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


class SemanticEnrichedMaxprobTotalDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_probs"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticEnrichedMaxprobTotalDissimilarityexp"
        else:
            base = "SemanticEnrichedMaxprobTotalDissimilarity"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_metrics = []  # To store enriched metrics for each sample

        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            # Step 1: Compute probabilities (negative log-probs)
            sample_probs = -np.array(sample_log_probs)
            if self.exp:
                sample_probs = -np.exp(-sample_probs)

            # Step 2: Compute row-wise sum of dissimilarities (1 - g)
            row_dissimilarities = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                sum_dissimilarities = np.sum(1 - row) - (1 - row[i])  # Exclude self-similarity
                row_dissimilarities.append(sum_dissimilarities)

            # Step 3: Normalize by (M - 1)
            normalized_dissimilarities = [
                dissim / (len(sample_sentence_similarity) - 1)
                for dissim in row_dissimilarities
            ]

            dissim = np.mean(normalized_dissimilarities)

            # Step 4: Enrich each metric
            enriched_sample_metrics = []
            for prob in sample_probs:
                enriched_metric = prob * dissim
                enriched_sample_metrics.append(enriched_metric)

            enriched_metrics.append(np.array(enriched_sample_metrics))

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_metrics):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


class SemanticEnrichedPPLAveDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,  
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_likelihoods"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticEnrichedPPLAveDissimilarityexp"
        else:
            base = "SemanticEnrichedPPLAveDissimilarity"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_ppl = []  # To store enriched PPL for each sample

        for sample_log_likelihoods, sample_sentence_similarity in zip(
            batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            # Step 1: Compute PPL for each sample
            ppl = -np.array([np.mean(token_ll) for token_ll in sample_log_likelihoods])
            if self.exp:
                ppl = -np.exp(-ppl)

            # Step 2: Compute row-wise average dissimilarity (1 - g)
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                # Compute average dissimilarity, excluding self-similarity
                average_dissimilarity = (np.sum(1 - row) - (1 - row[i])) / (len(row) - 1)
                row_averages.append(average_dissimilarity)

            # Step 3: Enrich each PPL independently by scaling with the average dissimilarity
            enriched_sample_ppl = []
            for i, (ppl_value, avg_dissimilarity) in enumerate(zip(ppl, row_averages)):
                if avg_dissimilarity == 0:
                    avg_dissimilarity = 1e-10  # Avoid division by zero
                enriched_value = ppl_value * avg_dissimilarity
                enriched_sample_ppl.append(enriched_value)

            enriched_ppl.append(np.array(enriched_sample_ppl))  # Collect enriched PPL values

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_ppl):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


class SemanticEnrichedPPLTotalDissimilarity(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        exp: bool = False,  
        sample_strategy: str = "first"
    ):
        super().__init__(
            ["sample_sentence_similarity", "sample_log_likelihoods"] + SAMPLE_SELECTION_STAT_KEYS,
            "sequence"
        )
        self.verbose = verbose
        self.exp = exp
        self.sample_strategy = sample_strategy

    def __str__(self):
        if self.exp:
            base = "SemanticEnrichedPPLTotalDissimilarityexp"
        else:
            base = "SemanticEnrichedPPLTotalDissimilarity"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_ppl = []  # To store enriched PPL for each sample

        for sample_log_likelihoods, sample_sentence_similarity in zip(
            batch_sample_log_likelihoods, batch_sample_sentence_similarity
        ):
            # Step 1: Compute PPL for each sample
            ppl = -np.array([np.mean(token_ll) for token_ll in sample_log_likelihoods])
            if self.exp:
                ppl = -np.exp(-ppl)

            # Step 2: Compute row-wise average dissimilarity (1 - g)
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                # Compute average dissimilarity, excluding self-similarity
                average_dissimilarity = (np.sum(1 - row) - (1 - row[i])) / (len(row) - 1)
                row_averages.append(average_dissimilarity)

            avg_dissimilarity = np.mean(row_averages)

            # Step 3: Enrich each PPL independently by scaling with the average dissimilarity
            enriched_sample_ppl = []
            for ppl_value in ppl:
                if avg_dissimilarity == 0:
                    avg_dissimilarity = 1e-10  # Avoid division by zero
                enriched_value = ppl_value * avg_dissimilarity
                enriched_sample_ppl.append(enriched_value)

            enriched_ppl.append(np.array(enriched_sample_ppl))  # Collect enriched PPL values

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_ppl):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


class SemanticEnrichedMTEAveDissimilarity(Estimator):
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
        return sample_strategy_to_prefix(self.sample_strategy) + "SemanticEnrichedMTEAveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_entropy = []

        for sample_entropy, sample_sentence_similarity in zip(
            batch_sample_entropy, batch_sample_sentence_similarity
        ):
            # Compute row-wise average dissimilarity (1 - g), excluding self-similarity
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_dissimilarity = (np.sum(1 - row) - (1 - row[i])) / (len(row) - 1)
                row_averages.append(average_dissimilarity)

            # Enrich each sample's entropy value
            enriched_sample_entropy = []
            for i, (entropy, avg_dissimilarity) in enumerate(zip(sample_entropy, row_averages)):
                if avg_dissimilarity == 0:
                    avg_dissimilarity = 1e-10  # Avoid division by zero
                enriched_value = entropy * avg_dissimilarity
                enriched_sample_entropy.append(enriched_value)

            enriched_entropy.append(np.array(enriched_sample_entropy))

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_entropy):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


class SemanticEnrichedMTETotalDissimilarity(Estimator):
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
        return sample_strategy_to_prefix(self.sample_strategy) + "SemanticEnrichedMTETotalDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_entropy = []

        for sample_entropy, sample_sentence_similarity in zip(
            batch_sample_entropy, batch_sample_sentence_similarity
        ):
            # Compute row-wise average dissimilarity (1 - g), excluding self-similarity
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_dissimilarity = (np.sum(1 - row) - (1 - row[i])) / (len(row) - 1)
                row_averages.append(average_dissimilarity)

            avg_dissimilarity = np.mean(row_averages)

            # Enrich each sample's entropy value
            enriched_sample_entropy = []
            for entropy in sample_entropy:
                if avg_dissimilarity == 0:
                    avg_dissimilarity = 1e-10  # Avoid division by zero
                enriched_value = entropy * avg_dissimilarity
                enriched_sample_entropy.append(enriched_value)

            enriched_entropy.append(np.array(enriched_sample_entropy))

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_entropy):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


class AveDissimilarity(Estimator):
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
        return sample_strategy_to_prefix(self.sample_strategy) + "AveDissimilarity"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_entropy = stats["sample_entropy"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        enriched_entropy = []

        for sample_entropy, sample_sentence_similarity in zip(
            batch_sample_entropy, batch_sample_sentence_similarity
        ):
            # Compute row-wise average dissimilarity (1 - g), excluding self-similarity
            row_averages = []
            for i in range(sample_sentence_similarity.shape[0]):
                row = sample_sentence_similarity[i]
                average_dissimilarity = (np.sum(1 - row) - (1 - row[i])) / (len(row) - 1)
                row_averages.append(average_dissimilarity)

            # Enrich each sample's entropy value
            enriched_sample_entropy = []
            for i, (entropy, avg_dissimilarity) in enumerate(zip(sample_entropy, row_averages)):
                if avg_dissimilarity == 0:
                    avg_dissimilarity = 1e-10  # Avoid division by zero
                enriched_value = avg_dissimilarity
                enriched_sample_entropy.append(enriched_value)

            enriched_entropy.append(np.array(enriched_sample_entropy))

        # Return only metric for the best sample for PRR calculation
        best_elements = []
        for best_id, metrics in zip(sample_ids, enriched_entropy):
            best_elements.append(metrics[best_id])

        return np.array(best_elements)


def load_estimator(cfg):
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "SemanticEnrichedMaxprobAveDissimilarity":
        return SemanticEnrichedMaxprobAveDissimilarity(**cfg)
    elif name == "SemanticEnrichedMaxprobTotalDissimilarity":
        return SemanticEnrichedMaxprobTotalDissimilarity(**cfg)
    elif name == "SemanticEnrichedPPLAveDissimilarity":
        return SemanticEnrichedPPLAveDissimilarity(**cfg)
    elif name == "SemanticEnrichedPPLTotalDissimilarity":
        return SemanticEnrichedPPLTotalDissimilarity(**cfg)
    elif name == "SemanticEnrichedMTEAveDissimilarity":
        return SemanticEnrichedMTEAveDissimilarity(**cfg)
    elif name == "SemanticEnrichedMTETotalDissimilarity":
        return SemanticEnrichedMTETotalDissimilarity(**cfg)
    elif name == "AveDissimilarity":
        return AveDissimilarity(**cfg)
    else:
        return None