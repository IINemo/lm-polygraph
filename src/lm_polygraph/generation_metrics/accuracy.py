import re
import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class AccuracyMetric(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    Two texts are considered equal if theis string representation is equal.
    """

    def __init__(self, remove_punctuation=True, normalize_texts=True):
        super().__init__(["greedy_texts"], "sequence")
        self.remove_punctuation = remove_punctuation
        self.normalize_texts = normalize_texts

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
        greedy_texts = stats["greedy_texts"]

        if self.normalize_texts:
            # gsm8k
            greedy_texts = [t.split("The answer is")[-1] for t in greedy_texts]
            target_texts = [t.split("\n####")[-1] for t in target_texts]

            # qa datasets
            greedy_texts = [t.replace("A:", "").split("Q:")[0] for t in greedy_texts]

            # mmlu
            greedy_texts = [
                (
                    re.search(r"\([A-Z]\)", t).group(0)
                    if (re.search(r"\([A-Z]\)", t) is not None)
                    else t
                )
                for t in greedy_texts
            ]

            # all datasets
            target_texts = [t.lower().strip() for t in target_texts]
            greedy_texts = [t.lower().strip() for t in greedy_texts]

        if self.remove_punctuation:
            greedy_texts = np.array([re.sub(r"[^\w\s]", "", t) for t in greedy_texts])
            target_texts = np.array([re.sub(r"[^\w\s]", "", t) for t in target_texts])

        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(greedy_texts, target_texts)
            ]
        )
