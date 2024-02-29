import numpy as np

from typing import Dict

from .estimator import Estimator
from scipy.special import softmax


class RenyiNeg(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "RenyiNeg" as provided in the paper https://arxiv.org/pdf/2212.09171.pdf.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation Rényi divergence between probability distribution for each token and uniform distribution.
    Code adapted from https://github.com/icannos/Todd/blob/master/Todd/itscorers.py
    """

    def __init__(
        self, verbose: bool = False, alpha: float = 0.5, temperature: float = 2
    ):
        super().__init__(["greedy_log_probs"], "sequence")
        self.verbose = verbose
        self.alpha = alpha
        self.temperature = temperature

    def __str__(self):
        return "RenyiNeg"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the Rényi divergence for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * logarithms of autoregressive probability distributions at each token in 'greedy_log_probs',
        Returns:
            np.ndarray: float Rényi divergence for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """

        batch_logits = stats["greedy_log_probs"]
        scores = []
        for logits in batch_logits:
            logits = np.array(logits)
            probabilities = softmax(logits / self.temperature, axis=-1)

            if self.alpha == 1:
                per_step_scores = np.log(probabilities) * probabilities
                per_step_scores = per_step_scores.sum(-1)
                per_step_scores += np.log(
                    np.ones_like(per_step_scores) * probabilities.shape[-1]
                )
            else:
                per_step_scores = np.log((probabilities**self.alpha).sum(-1))
                per_step_scores -= (self.alpha - 1) * np.log(
                    np.ones_like(per_step_scores) * probabilities.shape[-1]
                )
                per_step_scores *= 1 / (self.alpha - 1)
            scores.append(per_step_scores.mean(-1))

        return np.array(scores)
