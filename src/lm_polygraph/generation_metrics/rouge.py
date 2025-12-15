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
        # Ensure all targets and hypotheses are strings (handle lists defensively)
        def ensure_string(text):
            if isinstance(text, list):
                # If it's a list, take the first element
                if len(text) > 0:
                    text = text[0] if isinstance(text[0], str) else str(text[0])
                else:
                    text = ""
            elif not isinstance(text, str):
                text = str(text)
            return text
        
        processed_greedy_texts = [ensure_string(hyp) for hyp in stats["greedy_texts"]]
        processed_target_texts = [ensure_string(ref) for ref in target_texts]
        
        return np.array(
            [
                self._score_single(hyp, ref)
                for hyp, ref in zip(processed_greedy_texts, processed_target_texts)
            ]
        )
