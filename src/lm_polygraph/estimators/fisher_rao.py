import numpy as np

from typing import Dict

from .estimator import Estimator
from scipy.special import softmax


class FisherRao(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "FisherRao" as provided in the paper https://arxiv.org/pdf/2212.09171.pdf.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation Fisher-Rao distance between probability distribution for each token and uniform distribution.
    Code adapted from https://github.com/icannos/Todd/blob/master/Todd/itscorers.py
    """

    def __init__(self, verbose: bool = False, temperature: float = 2):
        super().__init__(["greedy_log_probs"], "sequence")
        self.verbose = verbose
        self.temperature = temperature

    def __str__(self):
        return "FisherRao"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the Fisher-Rao distance for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * logarithms of autoregressive probability distributions at each token in 'greedy_log_probs',
        Returns:
            np.ndarray: float Fisher-Rao distance for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """

        batch_logits = stats["greedy_log_probs"]
        scores = []
        for logits in batch_logits:
            logits = np.array(logits)
            probabilities = softmax(logits / self.temperature, axis=-1)
            per_step_scores = (
                2
                / np.pi
                * np.arccos(
                    np.sqrt(probabilities).sum(-1)
                    * np.sqrt(1 / probabilities.shape[-1])
                )
            )
            scores.append(per_step_scores.mean(-1))

        return np.array(scores)
