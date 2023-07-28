import numpy as np
from typing import List, Dict

from .generation_metric import GenerationMetric

MODEL_SCORES = ['model_rh']

class ModelScoreTokenwiseMetric(GenerationMetric):
    def __init__(self, score: str):
        assert score in MODEL_SCORES
        self.score = score
        super().__init__(MODEL_SCORES, 'token')

    def __str__(self):
        return 'ModelScoreToken-' + self.score.split('_')[-1]

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
            target_tokens: List[List[int]],
    ) -> np.ndarray:
        return np.array([s for sample in stats[self.score] for s in sample[:-1]])


class ModelScoreSeqMetric(GenerationMetric):
    def __init__(self, score: str):
        assert score in MODEL_SCORES
        self.score = score
        super().__init__(MODEL_SCORES, 'sequence')

    def __str__(self):
        return 'ModelScoreSeq-' + self.score.split('_')[-1]

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
            target_tokens: List[List[int]],
    ) -> np.ndarray:
        return np.array([np.logaddexp.reduce(sample) / len(sample) for sample in stats[self.score]])
