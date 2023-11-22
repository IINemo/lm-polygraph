import numpy as np

from typing import List, Dict
from abc import ABC, abstractmethod


class GenerationMetric(ABC):
    """
    Abstract generation metric class, which measures ground-truth uncertainty by comparing
    model-generated text with dataset ground-truth text. This ground-truth uncertainty is further
    compared with different estimators' uncertainties in UEManager using ue_metrics.
    """

    def __init__(self, stats_dependencies: List[str], level: str):
        """
        Parameters:
            stats_dependencies (List[str]):
                listed statistics which need to be calculated and passed in __call__ method.
                Statistics should include only names of statistics registered in lm_polygraph/stat_calculators/__init__.py
            level (str): uncertainty estimation level. Possible values:
                * 'sequence': method should output GenerationMetric for each input sequence in __call__.
                * 'token': method should output GenerationMetric for each token in input sequence in __call__.
        """
        assert level in ["sequence", "token"]
        self.level = level
        self.stats_dependencies = stats_dependencies

    @abstractmethod
    def __str__(self):
        """
        Abstract method. Returns unique name of the generation metric.
        Class parameters which affect generation metric estimates should also be included in the unique name
        to diversify between generation metrics.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        """
        Abstract method. Measures ground-truth uncertainty by comparing model-generated
        statistics from `stats` with dataset ground-truth texts from `target-texts`.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which includes values from
                statistics calculators for each self.stat_dependencies,
            target_texts (List[str]): ground-truth texts,
            target_tokens (List[List[int]]): corresponding token splits for each target text.
        Returns:
            np.ndarray: list of float ground-truth uncertainties calculated by comparing input
                statistics with ground truth texts.
                Should be 1-dimensional (in case of token-level, generation metrics from different
                samples should be concatenated). Higher values should indicate more confident samples
                (more similar to ground truth texts).
        """
        raise Exception("Not implemented")
