import numpy as np

from typing import Dict

from .estimator import Estimator


class MonteCarloSequenceEntropy(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["sample_log_probs"], "sequence")

    def __str__(self):
        return "MonteCarloSequenceEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates generation entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log probabilities for each token in each sample, in 'sample_log_probs'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["sample_log_probs"]
        return np.array([-np.mean(lp) for lp in logprobs])
