import numpy as np
from typing import List, Dict

from generation_metrics.generation_metric import GenerationMetric

BART_SCORES = ['rh']


class BartScoreTokenwiseMetric(GenerationMetric):
    def __init__(self, score: str):
        assert score in BART_SCORES
        self.score = score
        super().__init__(BART_SCORES, 'token')

    def __str__(self):
        return 'BARTScoreToken-' + self.score

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
            target_tokens: List[List[int]],
    ) -> np.ndarray:
        return np.array([s for sample in stats[self.score] for s in sample[:-1]])


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
        return np.array([np.logaddexp.reduce(sample) / len(sample) for sample in stats[self.score]])
