import numpy as np
from typing import List, Dict

from .generation_metric import GenerationMetric


class ModelScoreTokenwiseMetric(GenerationMetric):
    """
    Calculates token-level ModelScore metric between model-generated texts and ground truth texts.
    For each ground-truth text `r` and model-generated text 'h', method measures
    log-probabilities of generation 'h' on prompt 'Paraphrase "{r}"'.
    """

    def __init__(self):
        super().__init__(["model_rh"], "token")

    def __str__(self):
        return "ModelScoreToken-rh"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates token-level ModelScore between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log-probabilities of generation on prompt 'Paraphrase "{target text}"', in 'model_rh'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: concatenated float Model Scores for each token in each input sample.
        """
        return np.array([s for sample in stats["model_rh"] for s in sample[:-1]])


class ModelScoreSeqMetric(GenerationMetric):
    """
    Calculates sequence-level ModelScore metric between model-generated texts and ground truth texts.
    For each ground-truth text `r` and model-generated text 'h', method measures
    sum log-probabilitiy of generation 'h' on prompt 'Paraphrase "{r}"' normalized by the `h` length.
    """

    def __init__(self):
        super().__init__(["model_rh"], "sequence")

    def __str__(self):
        return "ModelScoreSeq-rh"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates sequence-level ModelScore between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log-probabilities of generation on prompt 'Paraphrase "{target text}"', in 'model_rh'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: float Model Scores for each input sample.
        """
        return np.array(
            [np.logaddexp.reduce(sample) / len(sample) for sample in stats["model_rh"]]
        )
