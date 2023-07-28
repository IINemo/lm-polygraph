import numpy as np

from typing import List, Dict
from abc import ABC, abstractmethod


class GenerationMetric(ABC):
    def __init__(self, stats_dependencies: List[str], level: str):
        assert level in ["sequence", "token"]
        self.level = level
        self.stats_dependencies = stats_dependencies

    @abstractmethod
    def __str__(self):
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        raise Exception("Not implemented")
