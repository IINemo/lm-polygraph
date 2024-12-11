import numpy as np
from rouge_score import rouge_scorer

from typing import List, Dict
from .generation_metric import GenerationMetric

from absl import logging as absl_logging

# This prevents bullshit spam from rouge scorer
absl_logging.set_verbosity(absl_logging.WARNING)


class RougeMetric(GenerationMetric):
    """
    Calculates Rouge metric between model-generated texts and ground truth texts.
    """

    def __init__(self, rouge_name, sample: bool = False):
        """
        Parameters:
            rouge_name (str): rouge metric type. Possible values:
                * rouge1
                * rouge2
                * rougeL
        """
        if sample:
            super().__init__(["first_sample_texts"], "sequence")
        else:
            super().__init__(["greedy_texts"], "sequence")
        self.sample = sample
        self.rouge_name = rouge_name
        self.scorer = rouge_scorer.RougeScorer([rouge_name], use_stemmer=True)

    def __str__(self):
        if self.sample:
            return f"SampleRouge_{self.rouge_name}"
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
    ) -> np.ndarray:
        """
        Calculates Rouge score between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of Rouge Scores for each sample in input.
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
