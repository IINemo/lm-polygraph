import numpy as np

from typing import Dict

from .estimator import Estimator


class SAR(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Token SAR" as provided in the paper https://arxiv.org/abs/2307.01379.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the sum of corrected probability using tokenSAR of the generated text
    and text relevance relative to all other generations.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(
            [
                "sample_sentence_similarity",
                "sample_log_likelihoods",
                "sample_token_similarity",
            ],
            "sequence",
        )
        self.verbose = verbose
        self.t = 0.001

    def __str__(self):
        return "SAR"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the SAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) for each sample in 'sample_log_likelihoods'
                * similarity for each sample of the generated text and generated text without one token for each token in 'sample_token_similarity',
                * matrix with cross-encoder similarities in 'sample_sentence_similarity'
        Returns:
            np.ndarray: float SAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_token_similarity = stats["sample_token_similarity"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        SAR = []
        for batch_data in zip(
            batch_sample_log_likelihoods,
            batch_sample_token_similarity,
            batch_sample_sentence_similarity,
        ):
            sample_log_likelihoods = batch_data[0]
            sample_token_similarity = batch_data[1]
            sample_sentence_similarity = batch_data[2]

            tokenSAR = []
            for log_likelihoods, token_similarity in zip(
                sample_log_likelihoods, sample_token_similarity
            ):
                log_likelihoods = np.array(log_likelihoods)
                R_t = 1 - token_similarity
                R_t_norm = R_t / R_t.sum()
                E_t = -log_likelihoods * R_t_norm
                tokenSAR.append(E_t.sum())

            tokenSAR = np.array(tokenSAR)
            probs_token_sar = np.exp(-tokenSAR)
            R_s = (
                probs_token_sar
                * sample_sentence_similarity
                * (1 - np.eye(sample_sentence_similarity.shape[0]))
            )
            sent_relevance = R_s.sum(-1) / self.t
            E_s = -np.log(sent_relevance + probs_token_sar)
            SAR.append(E_s.mean())
        return np.array(SAR)
