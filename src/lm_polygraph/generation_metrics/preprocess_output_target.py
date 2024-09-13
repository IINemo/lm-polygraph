import numpy as np

from copy import deepcopy
from typing import List, Dict
from .generation_metric import GenerationMetric


class PreprocessOutputTarget(GenerationMetric):
    """
    Preprocesses output and target texts before passing them to the base metric.
    """

    def __init__(self, base_metric, process_output_fn, process_target_fn):
        self.base_metric = getattr(base_metric, "base_metric", base_metric)
        self.level = base_metric.level
        self.stats_dependencies = base_metric.stats_dependencies
        self.process_output_fn = process_output_fn
        self.process_target_fn = process_target_fn

    def __str__(self):
        return str(self.base_metric)

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Applies preprocess functions to stats['greedy_texts'] and target_texts before passing them to the base metric.

        Parameters:
            stats (Dict[str, np.ndarray]): calculated stats
            target_texts (List[str]): ground-truth texts
            target_tokens (List[List[int]]): corresponding token splits for each target text
        Returns:
            np.ndarray: list of base metric values for each sample in input.
        """
        processed_target_texts = [
            self.process_target_fn(target) for target in target_texts
        ]

        # Select and copy only the stats that are needed for the base metric
        # before mutating greedy_texts with process_output_fn
        stats_copy = {k: v for k, v in stats.items() if k in self.stats_dependencies}
        stats_copy = deepcopy(stats_copy)

        stats_copy["greedy_texts"] = [
            self.process_output_fn(output) for output in stats_copy["greedy_texts"]
        ]

        return self.base_metric(stats_copy, processed_target_texts)
