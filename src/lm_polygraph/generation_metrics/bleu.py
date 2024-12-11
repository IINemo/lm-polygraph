import numpy as np
from sacrebleu.metrics import BLEU

from typing import List, Dict
from .generation_metric import GenerationMetric


class BLEUMetric(GenerationMetric):
    """
    Calculates BLEU metric between model-generated texts and ground truth texts.
    """

    def __init__(self, sample: bool = False):
        if sample:
            super().__init__(["first_sample_texts"], "sequence")
        else:
            super().__init__(["greedy_texts"], "sequence")
        self.sample = sample
        self.scorer = BLEU(effective_order=True, lowercase=True)

    def __str__(self):
        if self.sample:
            return "SampleBLEU"
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
        if self.sample:
            gen_texts = stats["first_sample_texts"]
        else:
            gen_texts = stats["greedy_texts"]

        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(gen_texts, target_texts)
            ]
        )
