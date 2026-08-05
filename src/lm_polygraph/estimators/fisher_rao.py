import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from scipy.special import softmax


class FisherRao(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "FisherRao" as provided in the paper https://arxiv.org/pdf/2212.09171.pdf.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation Fisher-Rao distance between probability distribution for each token and uniform distribution.
    By default, the distance is negated to follow the LM-Polygraph convention that
    higher scores indicate greater uncertainty. Set ``score_type="rainproof"`` to
    return the original RAINPROOF anomaly-score orientation, where higher scores
    indicate a distribution farther from uniform.
    Code adapted from https://github.com/icannos/Todd/blob/master/Todd/itscorers.py
    """

    def __init__(
        self,
        verbose: bool = False,
        temperature: float = 2,
        score_type: Literal["rainproof", "uncertainty"] = "uncertainty",
    ):
        if score_type not in ("rainproof", "uncertainty"):
            raise ValueError(
                "score_type must be either 'rainproof' or 'uncertainty', "
                f"got {score_type!r}"
            )
        super().__init__(["greedy_log_probs"], "sequence")
        self.verbose = verbose
        self.temperature = temperature
        self.score_type = score_type

    def __str__(self):
        return f"FisherRao_{self.score_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates a Fisher-Rao-based score for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * logarithms of autoregressive probability distributions at each token in 'greedy_log_probs',
        Returns:
            np.ndarray: mean token-level score for each sample in input statistics.
                With ``score_type="uncertainty"``, returns the negative Fisher-Rao
                distance, so higher values indicate more uncertain samples.
                With ``score_type="rainproof"``, returns the Fisher-Rao distance,
                so higher values indicate distributions farther from uniform.
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

        scores = np.array(scores)
        if self.score_type == "rainproof":
            return scores
        return -scores
