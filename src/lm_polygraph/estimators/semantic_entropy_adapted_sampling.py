import numpy as np

from typing import Dict

from .semantic_entropy import SemanticEntropy, Estimator


class SemanticEntropyAdaptedSampling(Estimator):
    def __init__(self, deberta_path: str = "microsoft/deberta-large-mnli"):
        super().__init__(
            [
                "adapted_sample_log_probs",
                "adapted_sample_log_probs_gen",
                "adapted_sample_texts",
            ],
            "sequence",
        )
        self.semantic_entropy = SemanticEntropy(deberta_path)

    def __str__(self):
        return "SemanticEntropyAdaptedSampling"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_p = stats["adapted_sample_log_probs"]
        log_p_gen = stats["adapted_sample_log_probs_gen"]
        texts = stats["adapted_sample_texts"]
        return self.semantic_entropy.batched_call(
            texts, log_p, log_weights=np.array(log_p) - np.array(log_p_gen)
        )
