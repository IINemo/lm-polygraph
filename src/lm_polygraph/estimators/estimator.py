import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict


class Estimator(ABC):
    def __init__(self, stats_dependencies: List[str], level: str):
        assert level in ["sequence", "token"]
        self.level = level
        self.stats_dependencies = stats_dependencies

    @abstractmethod
    def __str__(self):
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        raise Exception("Not implemented")
