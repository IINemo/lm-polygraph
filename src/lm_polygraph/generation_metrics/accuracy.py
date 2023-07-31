import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class AccuracyMetric(GenerationMetric):
    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return f"Accuracy"

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
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(stats["greedy_texts"], target_texts)
            ]
        )
