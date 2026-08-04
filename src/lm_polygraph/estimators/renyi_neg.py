import numpy as np

from typing import Dict, Literal

from .estimator import Estimator
from scipy.special import softmax


class RenyiNeg(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "RenyiNeg" as provided in the paper https://arxiv.org/pdf/2212.09171.pdf.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation Rényi divergence between probability distribution for each token and uniform distribution.
    By default, the divergence is negated to follow the LM-Polygraph convention that
    higher scores indicate greater uncertainty. Set ``score_type="rainproof"`` to
    return the original RAINPROOF anomaly-score orientation, where higher scores
    indicate a distribution farther from uniform.
    Code adapted from https://github.com/icannos/Todd/blob/master/Todd/itscorers.py
    """

    def __init__(
        self,
        verbose: bool = False,
        alpha: float = 0.5,
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
        self.alpha = alpha
        self.temperature = temperature
        self.score_type = score_type

    def __str__(self):
        return f"RenyiNeg_{self.score_type}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates a Rényi-based score for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * logarithms of autoregressive probability distributions at each token in 'greedy_log_probs',
        Returns:
            np.ndarray: mean token-level score for each sample in input statistics.
                With ``score_type="uncertainty"``, returns the negative Rényi
                divergence, so higher values indicate more uncertain samples.
                With ``score_type="rainproof"``, returns the Rényi divergence,
                so higher values indicate distributions farther from uniform.
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
                per_step_scores += (self.alpha - 1) * np.log(
                    np.ones_like(per_step_scores) * probabilities.shape[-1]
                )
                per_step_scores *= 1 / (self.alpha - 1)
            scores.append(per_step_scores.mean(-1))

        scores = np.array(scores)
        if self.score_type == "rainproof":
            return scores
        return -scores
