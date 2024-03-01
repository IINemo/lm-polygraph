import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class AccuracyMetric(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    Two texts are considered equal if theis string representation is equal.
    """

    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return "Accuracy"

    def _score_single(self, t1: str, t2: str) -> int:
        if t1.strip() == t2.strip():
            return 1
        return 0

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates accuracy between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of accuracies: 1 if generated text is equal to ground-truth and 0 otherwise.
        """
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(stats["greedy_texts"], target_texts)
            ]
        )
