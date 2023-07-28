import numpy as np

from typing import List, Dict
from abc import ABC, abstractmethod
from lm_polygraph.utils.model import WhiteboxModel

STAT_CALCULATORS: Dict[str, "StatCalculator"] = {}
STAT_DEPENDENCIES: Dict[str, List[str]] = {}


class StatCalculator(ABC):
    def __init__(self, stats: List[str], stat_dependencies: List[str]):
        self._stats = stats
        self._stat_dependencies = stat_dependencies

    @abstractmethod
    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, **kwargs) -> Dict[str, np.ndarray]:
        raise Exception('Not implemented')

    @property
    def stats(self) -> List[str]:
        return self._stats

    @property
    def stat_dependencies(self) -> List[str]:
        return self._stat_dependencies


def register(calculator_class: StatCalculator):
    for stat in calculator_class.stats:
        if stat in STAT_CALCULATORS.keys():
            continue
        STAT_CALCULATORS[stat] = calculator_class
        STAT_DEPENDENCIES[stat] = calculator_class.stat_dependencies
