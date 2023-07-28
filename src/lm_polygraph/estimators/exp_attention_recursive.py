import numpy as np

from typing import Dict

from .estimator import Estimator


class ExponentialAttentionRecursiveSeq(Estimator):
    def __init__(self, coef: float):
        self.coef = coef
        super().__init__(["entropy"], "sequence")

    def __str__(self):
        return f"ExponentialAttentionRecursiveSeq_{self.coef}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["entropy"]
        ue = []
        for entropy in entropies:
            entropy = np.array(entropy)
            attn = np.zeros(shape=(len(entropy), len(entropy)))
            for i in range(len(entropy)):
                for j in range(i):
                    attn[i, j] = self.coef ** (i - j)
            assert len(attn) == len(entropy)
            attn[1:] /= attn[1:].sum(1)[:, None]
            u = [entropy[0]]
            for i in range(1, len(entropy)):
                u.append(0.5 * entropy[i] + 0.5 * (attn[i - 1, :i] * np.array(u)).sum())
            ue.append(np.mean(u))
        return np.array(ue)


class ExponentialAttentionRecursiveToken(Estimator):
    def __init__(self, coef: float):
        self.coef = coef
        super().__init__(["entropy"], "token")

    def __str__(self):
        return f"AttentionRecursiveToken_{self.coef}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["entropy"]
        ue = []
        for entropy in entropies:
            entropy = np.array(entropy)
            attn = np.zeros(shape=(len(entropy), len(entropy)))
            for i in range(len(entropy)):
                for j in range(i):
                    attn[i, j] = self.coef ** (i - j)
            assert len(attn) == len(entropy)
            attn[1:] /= attn[1:].sum(1)[:, None]
            u = [entropy[0]]
            for i in range(1, len(entropy)):
                u.append(0.5 * entropy[i] + 0.5 * (attn[i - 1, :i] * np.array(u)).sum())
            ue.append(u[:-1])
        return np.concatenate(ue)
