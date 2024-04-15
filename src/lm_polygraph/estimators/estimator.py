import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict


class Estimator(ABC):
    """
    Abstract estimator class, which estimates the uncertainty of a language model.
    """

    def __init__(self, stats_dependencies: List[str], level: str):
        """
        Parameters:
            stats_dependencies (List[str]):
                listed statistics which need to be calculated and passed in __call__ method.
                Statistics should include only names of statistics registered in lm_polygraph/stat_calculators/__init__.py
            level (str): uncertainty estimation level. Possible values:
                * 'sequence': method estimates uncertainty (single float) for the whole model generation.
                * 'token': method estimates uncertainty for each token in the model generation.
        """
        assert level in ["sequence", "token"]
        self.level = level
        self.stats_dependencies = stats_dependencies

    @abstractmethod
    def __str__(self):
        """
        Abstract method. Returns unique name of the estimator.
        Class parameters which affect uncertainty estimates should also be included in the unique name
        to diversify between estimators.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Abstract method. Calculates the uncertainty for each text in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which includes values from
                statistics calculators for each `stat_dependencies`.
        Returns:
            np.ndarray: list of float uncertainties calculated for the input statistics.
                Should be 1-dimensional (in case of token-level, estimations from different
                samples should be concatenated). Higher values should indicate more uncertain samples.
        """
        raise Exception("Not implemented")
