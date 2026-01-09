import numpy as np
import torch
from typing import Dict
from .estimator import Estimator
from boostedprob import calculate_boostedprob


class BoostedProbSequence(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model by taking the average of the token-level scores obtained
    with BoostedProb (https://aclanthology.org/2025.emnlp-main.166.pdf)
    """

    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "BoostedProbSequence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the average of boosted model probabilities over the sequence

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * Full distribution of log p(y_i | y_<i, x) in 'greedy_log_probs'
        Returns:
            np.ndarray: average boosted model probabilities over each sequence sample.
                Higher values indicate more uncertain samples.
        """
        lprob_distributions = stats[
            "greedy_log_probs"
        ]  # nr_samples (nr_tokens x vocab_size)
        output_tokens = stats["greedy_tokens"]
        score = [
            calculate_boostedprob(torch.tensor(lprob_distribution), torch.tensor(out))
            for lprob_distribution, out in zip(lprob_distributions, output_tokens)
        ]

        return np.array([-np.mean(x.numpy()) for x in score])
