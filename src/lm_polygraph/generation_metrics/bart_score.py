import numpy as np
from typing import List, Dict

from .generation_metric import GenerationMetric

BART_SCORES = ['rh']

class BartScoreSeqMetric(GenerationMetric):
    def __init__(self, score: str):
        assert score in BART_SCORES
        self.score = score
        super().__init__(BART_SCORES, 'sequence')

    def __str__(self):
        return 'BARTScoreSeq-' + self.score

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
            target_tokens: List[List[int]],
    ) -> np.ndarray:
        return np.array(stats[self.score])
