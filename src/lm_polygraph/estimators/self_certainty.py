import numpy as np
from typing import Dict
from .estimator import Estimator


class SelfCertainty(Estimator):
    """
    Computes a self-certainty metric for language model outputs by estimating
    the KL divergence between a uniform distribution and the model's autoregressive
    token distribution at each position. Returns the negative mean of these divergences.
    A higher output value indicates higher uncertainty in the model's predictions.

    Reference:
        "Scalable Best-of-N Selection for Large Language Models via Self-Certainty"
        (https://arxiv.org/pdf/2502.18581)
    """

    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "SelfCertainty"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes the self-certainty score for each sample in the batch.
        For each token in a sample, calculates the KL divergence from a uniform
        distribution to the predicted distribution (approximated via negative log-prob),
        then averages across tokens. Finally, returns the negative mean of these
        per-token values per sample.

        Parameters:
            stats (Dict[str, np.ndarray]): A dictionary containing:
                - 'greedy_log_probs': list of list of np.ndarrays, where each inner array
                  contains log-probabilities over the vocabulary for a specific token position.

        Returns:
            np.ndarray: An array of self-certainty scores (one per sample).
                        Higher values mean higher uncertainty.
        """
        logprobs = stats["greedy_log_probs"]
        self_certainties: list[list[float]] = []

        for s_lp in logprobs:  # iterate over samples
            self_certainties.append([])
            for lp in s_lp:  # iterate over tokens in the sample
                token_logprobs = np.array(lp[~np.isinf(lp)])  # remove -inf values
                # Compute self-certainty: KL(uniform || predicted) = -mean(log p) - log(V)
                self_certainties[-1].append(
                    -np.mean(token_logprobs) - np.log(len(token_logprobs))
                )

        # Aggregate self-certainty over tokens, and negate to produce final score
        return np.array([-np.mean(sc) for sc in self_certainties])
