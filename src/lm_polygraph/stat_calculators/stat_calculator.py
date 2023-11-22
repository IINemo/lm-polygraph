import numpy as np

from typing import List, Dict
from abc import ABC, abstractmethod
from lm_polygraph.utils.model import Model

STAT_CALCULATORS: Dict[str, "StatCalculator"] = {}
STAT_DEPENDENCIES: Dict[str, List[str]] = {}


class StatCalculator(ABC):
    """
    Abstract class for some particular statistics calculation. Used to re-use same statistics across different
    uncertainty estimators at `lm_polygraph.estimators`. See the list of available calculators at
    lm_polygraph/stat_calculators/__init__.py.

    While estimators specify `stats_dependencies` to re-use these StatCalculator calculations, calculators can
    also specify dependencies on other calculators.

    UEManager at lm_polygraph.utils.manager will order all the needed calculators and estimators to be called in
    the correct order. Any cycle dependencies among calculators will be spotted by UEManager and end with an exception.

    Each new StatCalculator needs to be registered at lm_polygraph/stat_calculators/__init__.py to be seen be UEManager.
    """

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
        texts: List[str],
        model: Model,
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


def register(calculator_class: StatCalculator):
    """
    Registers a new statistics calculator to be seen by UEManager for properly organizing the calculations order.
    Needs to be called at lm_polygraph/stat_calculators/__init__.py for all stat calculators used in running benchmarks.
    """
    for stat in calculator_class.stats:
        if stat in STAT_CALCULATORS.keys():
            continue
        STAT_CALCULATORS[stat] = calculator_class
        STAT_DEPENDENCIES[stat] = calculator_class.stat_dependencies
