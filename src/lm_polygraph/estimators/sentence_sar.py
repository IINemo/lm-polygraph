import numpy as np

from typing import Dict

from .estimator import Estimator


class SentenceSAR(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Sentence SAR" as provided in the paper https://arxiv.org/abs/2307.01379.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the sum of the probability of the generated text and text relevance relative to all other generations.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(["sample_sentence_similarity", "sample_log_probs"], "sequence")
        self.verbose = verbose
        self.t = 0.001

    def __str__(self):
        return "SentenceSAR"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the sentenceSAR for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with cross-encoder similarities in 'sample_sentence_similarity'
        Returns:
            np.ndarray: float sentenceSAR for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]

        sentenceSAR = []
        for sample_log_probs, sample_sentence_similarity in zip(
            batch_sample_log_probs, batch_sample_sentence_similarity
        ):
            sample_probs = np.exp(np.array(sample_log_probs))
            R_s = (
                sample_probs
                * sample_sentence_similarity
                * (1 - np.eye(sample_sentence_similarity.shape[0]))
            )
            sent_relevance = R_s.sum(-1) / self.t
            E_s = -np.log(sent_relevance + sample_probs)
            sentenceSAR.append(E_s.mean())

        return np.array(sentenceSAR)
