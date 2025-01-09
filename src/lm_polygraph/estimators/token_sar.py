import numpy as np

from typing import Dict

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids


class TokenSAR(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Token SAR" as provided in the paper https://arxiv.org/abs/2307.01379.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the weighted sum of log_likelihoods with weights computed using token relevance.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(["token_similarity", "greedy_log_likelihoods"], "sequence")
        self.verbose = verbose

    def __str__(self):
        return "TokenSAR"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the tokenSAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
                * similarity of the generated text and generated text without one token for each token in 'token_similarity',
        Returns:
            np.ndarray: float tokenSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_log_likelihoods = stats["greedy_log_likelihoods"]
        batch_token_similarity = stats["token_similarity"]

        tokenSAR = []
        for log_likelihoods, token_similarity in zip(
            batch_log_likelihoods, batch_token_similarity
        ):
            log_likelihoods = np.array(log_likelihoods)
            R_t = 1 - token_similarity
            R_t_norm = R_t / R_t.sum()
            E_t = -log_likelihoods * R_t_norm
            tokenSAR.append(E_t.sum())

        return np.array(tokenSAR)


class SampledTokenSAR(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Token SAR" as provided in the paper https://arxiv.org/abs/2307.01379.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the weighted sum of log_likelihoods with weights computed using token relevance.
    """

    def __init__(self, verbose: bool = False, sample_strategy: str = "first"):
        super().__init__(["sample_token_similarity", "sample_log_likelihoods"], "sequence")
        self.verbose = verbose
        self.sample_strategy = sample_strategy

    def __str__(self):
        return sample_strategy_to_prefix(self.sample_strategy) + "SampledTokenSAR"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the tokenSAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
                * similarity of the generated text and generated text without one token for each token in 'token_similarity',
        Returns:
            np.ndarray: float tokenSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        sample_ids = best_sample_ids(self.sample_strategy, stats)

        result = []
        for batch_data in zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            sample_ids,
        ):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            best_id = batch_data[2]

            tokenSAR = []
            for log_likelihoods, token_similarity in zip(
                sample_log_likelihoods, sample_token_similarity
            ):
                log_likelihoods = np.array(log_likelihoods)
                R_t = 1 - token_similarity
                R_t_norm = R_t / R_t.sum()
                E_t = -log_likelihoods * R_t_norm
                tokenSAR.append(E_t.sum())
            result.append(tokenSAR[best_id])

        return np.array(result)
