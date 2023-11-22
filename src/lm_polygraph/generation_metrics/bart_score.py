import numpy as np
from typing import List, Dict

from .generation_metric import GenerationMetric

BART_SCORES = ["rh"]


class BartScoreSeqMetric(GenerationMetric):
    """
    Calculates BARTScore metric (https://arxiv.org/abs/2106.11520)
    between model-generated texts and ground truth texts.
    """

    def __init__(self, score: str):
        assert score in BART_SCORES
        self.score = score
        super().__init__(BART_SCORES, "sequence")

    def __str__(self):
        return "BARTScoreSeq-" + self.score

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates BARTScore(https://arxiv.org/abs/2106.11520) between
        stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of BART Scores for each sample in input.
        """
        return np.array(stats[self.score])
