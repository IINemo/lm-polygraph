import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.process_probs import process_probs

class CocoaMSP(Estimator):
    def __init__(
        self,
        samples_source: str = "sample"
    ):
        super().__init__(
            [f"greedy_{samples_source}_sentence_similarity", "greedy_log_likelihoods"], "sequence"
        )
        self.samples_source = samples_source

    def __str__(self):
        base = "CocoaMSP"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]
        batch_lls = np.array(
            [
                np.sum(log_likelihood)
                for log_likelihood in stats["greedy_log_likelihoods"]
            ]
        )

        enriched_metrics = []  # To store enriched metrics for each sample
        for greedy_ll, greedy_sentence_similarity in zip(
            batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            avg_dissimilarity = np.mean(1 - np.array(greedy_sentence_similarity))

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class CocoaMSPP(Estimator):
    def __init__(
        self,
        samples_source: str = "sample",
        **process_probs_args,
    ):
        super().__init__(
            [
                f"greedy_{samples_source}_sentence_similarity",
                f"{samples_source}_log_likelihoods",
                "greedy_log_likelihoods",
            ], "sequence"
        )
        self.samples_source = samples_source
        self.process_probs_args = process_probs_args

    def __str__(self):
        base = "CocoaMSPP"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]
        batch_sample_token_lls = stats[f"{self.samples_source}_log_likelihoods"]
        batch_lls = np.array(
            [
                np.sum(log_likelihood)
                for log_likelihood in stats["greedy_log_likelihoods"]
            ]
        )

        enriched_metrics = []  # To store enriched metrics for each sample
        for sample_token_lls, greedy_ll, greedy_sentence_similarity in zip(
            batch_sample_token_lls, batch_lls, batch_greedy_sentence_similarity
        ):
            # Compute probabilities (negative log-probs)
            prob = -greedy_ll

            # Compute row-wise average similarity, excluding self-similarity
            # Diagonal contains self-similarities
            d = 1 - np.array(greedy_sentence_similarity)
            probs = np.array([np.exp(sum(x)) for x in sample_token_lls])
            probs = process_probs(probs, **self.process_probs_args)
            avg_dissimilarity = (d * probs).sum()

            enriched_metric = prob * avg_dissimilarity
            enriched_metrics.append(enriched_metric)

        return np.array(enriched_metrics)


class CocoaPPL(Estimator):
    def __init__(
        self,
        samples_source: str = "sample"
    ):
        super().__init__(
            [f"greedy_{samples_source}_sentence_similarity", "greedy_log_likelihoods"], "sequence"
        )
        self.samples_source = samples_source

    def __str__(self):
        base = "CocoaPPL"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]

        enriched_ppl = []  # To store enriched PPL for each sample

        for greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)

            # Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - np.array(greedy_sentence_similarity))

            enriched_value = ppl * avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class CocoaPPLP(Estimator):
    def __init__(
        self,
        samples_source: str = "sample",
        **process_probs_args,
    ):
        super().__init__(
            [
                f"greedy_{samples_source}_sentence_similarity",
                f"{samples_source}_log_likelihoods",
                "greedy_log_likelihoods",
            ], "sequence"
        )
        self.samples_source = samples_source
        self.process_probs_args = process_probs_args

    def __str__(self):
        base = "CocoaPPLP"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]
        batch_sample_token_lls = stats[f"{self.samples_source}_log_likelihoods"]

        enriched_ppl = []  # To store enriched PPL for each sample

        for sample_token_lls, greedy_log_likelihoods, greedy_sentence_similarity in zip(
            batch_sample_token_lls, batch_greedy_log_likelihoods, batch_greedy_sentence_similarity
        ):
            # get PPL for each sample
            ppl = -np.mean(greedy_log_likelihoods)

            # Compute row-wise average similarity, excluding self-similarity
            d = 1 - np.array(greedy_sentence_similarity)
            probs = np.array([np.exp(sum(x)) for x in sample_token_lls])
            probs = process_probs(probs, **self.process_probs_args)
            avg_dissimilarity = (d * probs).sum()

            enriched_value = ppl * avg_dissimilarity
            enriched_ppl.append(enriched_value)

        return np.array(enriched_ppl)


class CocoaMTE(Estimator):
    def __init__(
        self,
        samples_source: str = "sample"
    ):
        super().__init__([f"greedy_{samples_source}_sentence_similarity", "entropy"], "sequence")
        self.samples_source = samples_source

    def __str__(self):
        base = "CocoaMTE"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]

        enriched_entropy = []

        for greedy_entropy, greedy_sentence_similarity in zip(
            batch_greedy_entropy, batch_greedy_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            avg_dissimilarity = np.mean(1 - np.array(greedy_sentence_similarity))

            entropy = np.mean(greedy_entropy)
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)


class CocoaMTEP(Estimator):
    def __init__(
        self,
        samples_source: str = "beamsearch",
        **process_probs_args,
    ):
        super().__init__([
            f"greedy_{samples_source}_sentence_similarity",
            f"{samples_source}_log_likelihoods",
            "entropy",
        ], "sequence")
        self.samples_source = samples_source
        self.process_probs_args = process_probs_args

    def __str__(self):
        base = "CocoaMTEP"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_entropy = stats["entropy"]
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]
        batch_sample_token_lls = stats[f"{self.samples_source}_log_likelihoods"]

        enriched_entropy = []

        for sample_token_lls, greedy_entropy, greedy_sentence_similarity in zip(
            batch_sample_token_lls, batch_greedy_entropy, batch_greedy_sentence_similarity
        ):
            #  Compute row-wise average similarity, excluding self-similarity
            d = 1 - np.array(greedy_sentence_similarity)
            probs = np.array([np.exp(sum(x)) for x in sample_token_lls])
            probs = process_probs(probs, **self.process_probs_args)
            avg_dissimilarity = (d * probs).sum()

            entropy = np.mean(greedy_entropy)
            enriched_value = entropy * avg_dissimilarity
            enriched_entropy.append(enriched_value)

        return np.array(enriched_entropy)


class Dissimilarity(Estimator):
    def __init__(
            self,
            samples_source: str = "sample"
    ):
        super().__init__([f"greedy_{samples_source}_sentence_similarity"], "sequence")
        self.samples_source = samples_source

    def __str__(self):
        base = "Dissimilarity"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]

        dissims = []

        for greedy_sentence_similarity in batch_greedy_sentence_similarity:
            dissims.append(np.mean(1 - np.array(greedy_sentence_similarity)))

        return np.array(dissims)


class DissimilarityP(Estimator):
    def __init__(
            self,
            samples_source: str = "beamsearch",
            **process_probs_args,
    ):
        super().__init__([
            f"greedy_{samples_source}_sentence_similarity",
            f"{samples_source}_log_likelihoods",
        ], "sequence")
        self.samples_source = samples_source
        self.process_probs_args = process_probs_args

    def __str__(self):
        base = "DissimilarityP"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        return base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_greedy_sentence_similarity = [x[0][1:] for x in stats[f"greedy+{self.samples_source}_semantic_matrix_entail"]]
        batch_sample_token_lls = stats[f"{self.samples_source}_log_likelihoods"]

        dissims = []

        for sample_token_lls, greedy_sentence_similarity in zip(
                batch_sample_token_lls,
                batch_greedy_sentence_similarity,
        ):
            d = 1 - np.array(greedy_sentence_similarity)
            probs = np.array([np.exp(sum(x)) for x in sample_token_lls])
            probs = process_probs(probs, **self.process_probs_args)
            dissims.append((d * probs).sum())

        return np.array(dissims)