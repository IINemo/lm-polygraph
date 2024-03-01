import numpy as np
from rouge_score import rouge_scorer

from typing import List, Dict
from .generation_metric import GenerationMetric


class RougeMetric(GenerationMetric):
    """
    Calculates Rouge metric between model-generated texts and ground truth texts.
    """

    def __init__(self, rouge_name):
        """
        Parameters:
            rouge_name (str): rouge metric type. Possible values:
                * rouge1
                * rouge2
                * rougeL
        """
        super().__init__(["greedy_texts"], "sequence")
        self.rouge_name = rouge_name
        self.scorer = rouge_scorer.RougeScorer([rouge_name], use_stemmer=True)

    def __str__(self):
        return f"Rouge_{self.rouge_name}"

    def _score_single(self, t1: str, t2: str):
        sc = self.scorer.score(t1, t2)[self.rouge_name].fmeasure
        sc_best = self.scorer.score(t2, t2)[self.rouge_name].fmeasure
        if sc_best == 0:
            return np.nan
        return sc / sc_best

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates Rouge score between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of Rouge Scores for each sample in input.
        """
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(stats["greedy_texts"], target_texts)
            ]
        )
