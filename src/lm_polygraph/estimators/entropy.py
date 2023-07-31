import numpy as np

from typing import Dict

from .estimator import Estimator


class EntropySeq(Estimator):
    def __init__(self):
        super().__init__(['entropy'], 'sequence')

    def __str__(self):
        return 'EntropySeq'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropy = stats['entropy']
        return np.array([np.mean(e) for e in entropy])


class EntropyToken(Estimator):
    def __init__(self):
        super().__init__(['entropy'], 'token')

    def __str__(self):
        return 'EntropyToken'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropy = stats['entropy']
        return np.concatenate([np.array(e[:-1]) for e in entropy])
