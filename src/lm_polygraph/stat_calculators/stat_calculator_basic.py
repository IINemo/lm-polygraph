import numpy as np

from typing import List, Dict
from abc import ABC, abstractmethod
from lm_polygraph.utils.model import Model


class StatCalculatorBasic(ABC):
    def __init__(self, stats: List[str], stat_dependencies: List[str]):
        """
        Parameters:
            stats: List[str]: Names of statiscits which can be calculated by using this StatCalculator.
            stat_dependencies: List[str]: Names of statistics which this calculator needs to use. Can be any names of
                other StatCalculators. Any cycle dependencies among calculators will be spotted by UEManager and
                end with an exception.
        """
        self._stats = stats
        self._stat_dependencies = stat_dependencies

    @abstractmethod
    def __call__(
        self,
        dependencies: Dict[str, np.array],
        model_inputs,
        model: Model,
        model_args,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Abstract method. Calculates the statistic based on the other provided statistics.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, which includes values from
                statistics calculators for each `stat_dependencies`.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with calculated statistics under all keys from `stats`.
        """
        raise Exception("Not implemented")
    
    @property
    def stats(self) -> List[str]:
        """
        Returns:
            List[str]: Names of statistics which can be calculated by this class.
        """
        return self._stats

    @property
    def stat_dependencies(self) -> List[str]:
        """
        Returns:
            List[str]: Names of statistics dependencies which this class needs at __call__.
        """
        return self._stat_dependencies
