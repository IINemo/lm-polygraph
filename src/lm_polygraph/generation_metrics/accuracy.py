import re
import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class AccuracyMetric(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    Two texts are considered equal if theis string representation is equal.
    """

    def __init__(self, target_ignore_regex = None, answer_ignore_regex = None):
        super().__init__(["greedy_texts"], "sequence")
        self.target_ignore_regex = re.compile(target_ignore_regex) if target_ignore_regex else None
        self.answer_ignore_regex = re.compile(answer_ignore_regex) if answer_ignore_regex else None

    def __str__(self):
        return "Accuracy"

    def _score_single(self, output: str, target: str) -> int:
        if output.strip() == target.strip():
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
        greedy_texts = stats["greedy_texts"]

        if self.target_ignore_regex:
            target_texts = [self.target_ignore_regex.sub("", t) for t in target_texts]
        if self.answer_ignore_regex:
            greedy_texts = [self.answer_ignore_regex.sub("", t) for t in greedy_texts]

        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(stats["greedy_texts"], target_texts)
            ]
        )
