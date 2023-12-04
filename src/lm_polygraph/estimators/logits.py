import numpy as np

from typing import Dict

from .estimator import Estimator


class MaximumSequenceLogitsSum(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model by calculating the
    average of logits of the generation with minus sign.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(["greedy_lm_logits"], "sequence")

    def __str__(self):
        return "MaximumSequenceLogitsSum"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the minus logits average of each sample in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log h(y_i | y_<i, x) in 'greedy_log_likelihoods', where h is the model output
        Returns:
            np.ndarray: minus logits for each sample.
                Higher values indicate more uncertain samples.
        """
        logits = stats["greedy_lm_logits"]
        return np.array([-np.sum(l) for l in logits])


class MaximumSequenceLogitsAverage(Estimator):
        """
        Estimates the sequence-level uncertainty of a language model by calculating the
        average of logits of the generation with minus sign.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
        """

        def __init__(self):
            super().__init__(["greedy_lm_logits"], "sequence")

        def __str__(self):
            return "MaximumSequenceLogitsAverage"

        def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
            """
            Estimates the minus logits average of each sample in input statistics.

            Parameters:
                stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                    * log h(y_i | y_<i, x) in 'greedy_log_likelihoods', where h is the model output
            Returns:
                np.ndarray: minus logits for each sample.
                    Higher values indicate more uncertain samples.
            """
            logits = stats["greedy_lm_logits"]
            return np.array([-np.mean(l) for l in logits])
