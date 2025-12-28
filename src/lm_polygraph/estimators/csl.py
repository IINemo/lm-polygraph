import numpy as np
from typing import Dict

from .estimator import Estimator


class CSL(Estimator):
    """
    CSL (Contextualized Sequence Likelihood) from https://aclanthology.org/2024.emnlp-main.578/

    This estimator quantifies uncertainty in LLM outputs by reweighting token logits
    """

    def __init__(
        self,
    ):
        super().__init__(
            ["greedy_log_likelihoods", "attention_weights_eliciting_prompt"], "sequence"
        )

    def __str__(self) -> str:
        return "CSL"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate uncertainty scores for each sequence in a batch.

        Args:
            stats: Dictionary containing model statistics including attention weights and log likelihoods

        Returns:
            np.ndarray: Uncertainty scores for each sequence
        """
        log_likelihoods = stats["greedy_log_likelihoods"]
        attention_weights = stats["attention_weights_eliciting_prompt"]

        csl_scores = []
        for i in range(len(log_likelihoods)):
            weights = attention_weights[i].mean(0)
            weights /= weights.sum()
            csl_scores.append(-np.sum(log_likelihoods[i] * weights))

        return np.array(csl_scores)
