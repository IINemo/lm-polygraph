import numpy as np
from sacrebleu.metrics import BLEU

from typing import List, Dict
from .generation_metric import GenerationMetric


class BLEUMetric(GenerationMetric):
    """
    Calculates BLEU metric between model-generated texts and ground truth texts.
    """

    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")
        self.scorer = BLEU(effective_order=True, lowercase=True)

    def __str__(self):
        return "BLEU"

    def _score_single(self, t1: str, t2: str):
        return self.scorer.sentence_score(
            t1.strip().rstrip("."), [t2.strip().rstrip(".")]
        ).score

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates BLEU score between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of BLEU Scores for each sample in input.
        """
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(stats["greedy_texts"], target_texts)
            ]
        )
