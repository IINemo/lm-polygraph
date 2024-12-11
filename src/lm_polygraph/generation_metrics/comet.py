import re
import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class Comet(GenerationMetric):
    """
    Calculates COMET metric (https://aclanthology.org/2020.emnlp-main.213/)
    between model-generated texts and ground truth texts.
    """

    def __init__(self, scorer, source_ignore_regex=None, lang="en", sample: bool = False):
        if sample:
            super().__init__(["first_sample_texts", "input_texts"], "sequence")
        else:
            super().__init__(["greedy_texts", "input_texts"], "sequence")
        self.sample = sample
        self.source_ignore_regex = (
            re.compile(source_ignore_regex) if source_ignore_regex else None
        )
        self.scorer = scorer

    def __str__(self):
        if self.sample:
            return "SampleComet"
        return "Comet"

    def _filter_text(self, text: str, ignore_regex: re.Pattern) -> str:
        if ignore_regex is not None:
            processed_text = ignore_regex.search(text)
            if processed_text:
                return processed_text.group(1)
            else:
                raise ValueError(
                    f"Source text {text} does not match the ignore regex {ignore_regex}"
                )
        return text

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates COMET (https://aclanthology.org/2020.emnlp-main.213/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            input_texts (List[str]): input texts before translation
        Returns:
            np.ndarray: list of COMET Scores for each sample in input.
        """
        sources = [
            self._filter_text(src, self.source_ignore_regex)
            for src in stats["input_texts"]
        ]

        if self.sample:
            gen_texts = stats["first_sample_texts"]
        else:
            gen_texts = stats["greedy_texts"]

        scores = np.array(
            self.scorer.compute(
                predictions=gen_texts,
                references=target_texts,
                sources=sources,
            )["scores"]
        )
        return scores
