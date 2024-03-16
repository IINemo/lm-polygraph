import numpy as np
from evaluate import load

from typing import List, Dict
from .generation_metric import GenerationMetric


class Comet(GenerationMetric):
    """
    Calculates COMET metric (https://aclanthology.org/2020.emnlp-main.213/)
    between model-generated texts and ground truth texts.
    """

    def __init__(self, lang="en"):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        self.scorer = load("comet")

    def __str__(self):
        return "Comet"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Calculates COMET (https://aclanthology.org/2020.emnlp-main.213/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            input_texts (List[str]): input texts before translation
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of COMET Scores for each sample in input.
        """
        # remove translation prompt
        sources = [
            s.split("translation:\n")[-1].replace("\nTranslation:\n", "")
            for s in stats["input_texts"]
        ]
        scores = np.array(
            self.scorer.compute(
                predictions=stats["greedy_texts"],
                references=target_texts,
                sources=sources,
            )["scores"]
        )
        return scores
