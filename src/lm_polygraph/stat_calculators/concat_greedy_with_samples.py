import numpy as np

import itertools
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel


class ConcatGreedyWithSamplesBase(StatCalculator):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [], []

    def __init__(
        self,
        sample_source: str = "sample",
    ):
        super().__init__()
        self.sample_source = sample_source

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        res = {}
        for stat in ['log_likelihoods', 'texts', 'tokens']:
            greedy_stats = dependencies[f"greedy_{stat}"]
            sample_stats = dependencies[f"{self.sample_source}_{stat}"]
            concat = [[g] + s for g, s in zip(greedy_stats, sample_stats)]
            res[f'greedy+{self.sample_source}_{stat}'] = concat
        return res


class ConcatGreedyWithSamples(ConcatGreedyWithSamplesBase):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "greedy+sample_log_likelihoods",
            "greedy+sample_texts",
            "greedy+sample_tokens",
        ], [
            "greedy_log_likelihoods",
            "greedy_texts",
            "greedy_tokens",
            "sample_log_likelihoods",
            "sample_texts",
            "sample_tokens",
        ]

    def __init__(self):
        super().__init__("sample")


class ConcatGreedyWithBeam(ConcatGreedyWithSamplesBase):
    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "greedy+beamsearch_log_likelihoods",
            "greedy+beamsearch_texts",
            "greedy+beamsearch_tokens",
        ], [
            "greedy_log_likelihoods",
            "greedy_texts",
            "greedy_tokens",
            "beamsearch_log_likelihoods",
            "beamsearch_texts",
            "beamsearch_tokens",
        ]

    def __init__(self):
        super().__init__("beamsearch")