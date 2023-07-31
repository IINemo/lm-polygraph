import numpy as np

from typing import Dict

from .estimator import Estimator


class ExponentialAttentionEntropySeq(Estimator):
    def __init__(self, coef: float):
        self.coef = coef
        super().__init__(['entropy'], 'sequence')

    def __str__(self):
        return f'ExponentialAttentionEntropySeq_{self.coef}'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats['entropy']
        ue = []
        for entropy in entropies:
            entropy = np.array(entropy)
            attn = np.zeros(shape=(len(entropy), len(entropy)))
            for i in range(len(entropy)):
                for j in range(i):
                    attn[i, j] = self.coef ** (i - j)
            attn[1:] /= attn[1:].sum(1)[:, None]
            np.fill_diagonal(attn, 1)
            attn /= attn.sum(1)[:, None]
            ue.append(np.mean(attn.sum(0) * entropy))
        return np.array(ue)


class ExponentialAttentionEntropyToken(Estimator):
    def __init__(self, coef: float):
        self.coef = coef
        super().__init__(['entropy'], 'token')

    def __str__(self):
        return f'ExponentialAttentionEntropyToken_{self.coef}'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats['entropy']
        ue = []
        for entropy in entropies:
            entropy = np.array(entropy)
            attn = np.zeros(shape=(len(entropy), len(entropy)))
            for i in range(len(entropy)):
                for j in range(i):
                    attn[i, j] = self.coef ** (i - j)
            attn[1:] /= attn[1:].sum(1)[:, None]
            np.fill_diagonal(attn, 1)
            attn /= attn.sum(1)[:, None]
            ue.append((attn.sum(0) * entropy)[:-1])
        return np.concatenate(ue)
