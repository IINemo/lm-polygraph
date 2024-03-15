import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class AccuracyMetric(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    Two texts are considered equal if theis string representation is equal.
    """

    def __init__(self, remove_punctuation=True):
        super().__init__(["greedy_texts"], "sequence")
        self.remove_punctuation = remove_punctuation

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
        
        if self.remove_punctuation:
            greedy_texts = np.array([re.sub(r'[^\w\s]','',t.replace("A: ", "").split("Q:")[0].lower()) for t in stats['greedy_texts']])
        else:
            greedy_texts = np.array(stats['greedy_texts'])
        
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(greedy_texts, target_texts)
            ]
        )
