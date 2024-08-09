import numpy as np
import re

from typing import Dict

from .estimator import Estimator


class Linguistic1S(Estimator):
    def __init__(self, expressions: Dict[str, float]):
        self.expressions = expressions
        super().__init__(['greedy_texts'], "sequence")

    def __str__(self):
        return f"Linguistic1S"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats["model"]

        for answer in stats["greedy_texts"]:
            ue = np.nan
            for expression, confidence in self.expressions.items():
                if expression in answer:
                    ue = 1 - confidence
                    break
            ues.append(ue)

        return np.array(ues)
