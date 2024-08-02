import numpy as np

from typing import Dict

from .estimator import Estimator


class Verbalized1S(Estimator):
    def __init__(self, topk=1, cot=False):
        if cot:
            super().__init__([f"verbalized_1s_cot_response"], "sequence")
        else:
            self.topk = topk
            super().__init__([f"verbalized_1s_top{topk}_response"], "sequence")

    def __str__(self):
        return f"Verbalized1STop{self.topk}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # parse the text
        return np.array([])
