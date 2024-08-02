import numpy as np

from typing import Dict

from .estimator import Estimator


class Verbalized2S(Estimator):
    def __init__(self, topk=1):
        self.topk = topk
        super().__init__([f"verbalized_2s_top{topk}_response"], "sequence")

    def __str__(self):
        return f"Verbalized2STop{self.topk}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # parse the text
        return np.array([])
