import numpy as np

from collections import defaultdict
from typing import List, Dict, Optional

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
        super().__init__(["greedy_logits"], "sequence")
        self.verbose = verbose
        self.temperature = temperature

    def __str__(self):
        return "FisherRao"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the Fisher-Rao distance for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'greedy_logits',
        Returns:
            np.ndarray: float Fisher-Rao distance for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """

        logits = np.array(stats["greedy_logits"])
        probabilities = softmax(logits / self.temperature, axis=-1)
        per_step_scores = (
            2
            / np.pi
            * np.arccos(
                np.sqrt(probabilities).sum(-1) * np.sqrt(1 / probabilities.shape[-1])
            )
        )

        return per_step_scores.mean(-1)
