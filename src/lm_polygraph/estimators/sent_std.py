import numpy as np

from typing import Dict

from .estimator import Estimator


class StdSeq(Estimator):
    def __init__(self):
        super().__init__(['std'], 'sequence')

    def __str__(self):
        return 'StdSeq'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        std = stats['std']
        return np.array([-np.mean(e) for e in std])


class StdToken(Estimator):
    def __init__(self):
        super().__init__(['std'], 'token')

    def __str__(self):
        return 'StdToken'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        std = stats['std']
        return np.concatenate([-np.array(e[:-1]) for e in std])
