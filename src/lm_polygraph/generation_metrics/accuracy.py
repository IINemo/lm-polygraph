import re
import string
import numpy as np
import logging

from typing import List, Dict
from .generation_metric import GenerationMetric

log = logging.getLogger("lm_polygraph")


class AccuracyMetric(GenerationMetric):
    """
    Calculates accuracy between model-generated texts and ground-truth.
    Two texts are considered equal if theis string representation is equal.
    """

    def __init__(
        self, target_ignore_regex=None, output_ignore_regex=None, normalize=False
    ):
        super().__init__(["greedy_texts"], "sequence")
        self.target_ignore_regex = (
            re.compile(target_ignore_regex) if target_ignore_regex else None
        )
        self.output_ignore_regex = (
            re.compile(output_ignore_regex) if output_ignore_regex else None
        )
        self.normalize = normalize

        if self.target_ignore_regex or self.output_ignore_regex or self.normalize:
            log.warning(
                "Specifying ignore_regex or normalize in AccuracyMetric is deprecated. Use output and target processing scripts instead."
            )

    def __str__(self):
        return "Accuracy"

    def _score_single(self, output: str, target: str) -> int:
        if output.strip() == target.strip():
            return 1
        return 0

    def _filter_text(self, text: str, ignore_regex: re.Pattern) -> str:
        text = ignore_regex.sub("", text) if ignore_regex else text

        return text

    def _normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        return text

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates accuracy between stats['greedy_texts'] and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
        Returns:
            np.ndarray: list of accuracies: 1 if generated text is equal to ground-truth and 0 otherwise.
        """
        greedy_texts = stats["greedy_texts"]

        result = []

        for hyp, ref in zip(greedy_texts, target_texts):
            ref = self._filter_text(ref, self.target_ignore_regex)
            hyp = self._filter_text(hyp, self.output_ignore_regex)

            if self.normalize:
                ref = self._normalize_text(ref)
                hyp = self._normalize_text(hyp)

            result.append(self._score_single(hyp, ref))

        return np.array(result)
