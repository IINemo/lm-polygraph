import numpy as np

from typing import Dict

from .estimator import Estimator


class Linguistic1S(Estimator):
    def __init__(self, expressions: Dict[str, float], name_postfix=""):
        self.expressions = expressions
        self.postfix = name_postfix
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return f"Linguistic1S{self.postfix}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ues = []
        for answer in stats["greedy_texts"]:
            ue = np.nan
            for expression, confidence in self.expressions.items():
                if expression in answer:
                    ue = 1 - confidence
                    break
            ues.append(ue)

        return np.array(ues)
