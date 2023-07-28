import numpy as np

from typing import Dict

from .estimator import Estimator


class AttentionRecursiveSeq(Estimator):
    def __init__(self):
        super().__init__(["entropy", "attention"], "sequence")

    def __str__(self):
        return "AttentionRecursiveSeq"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["entropy"]
        attentions = stats["attention"]
        ue = []
        for attn, entropy in zip(attentions, entropies):
            entropy = np.array(entropy)
            assert len(attn) == len(entropy)
            attn[1:] /= attn[1:].sum(1)[:, None]
            u = [entropy[0]]
            for i in range(1, len(entropy)):
                u.append(0.5 * entropy[i] + 0.5 * (attn[i - 1, :i] * np.array(u)).sum())
            ue.append(np.mean(u))
        return np.array(ue)


class AttentionRecursiveToken(Estimator):
    def __init__(self):
        super().__init__(["entropy", "attention"], "token")

    def __str__(self):
        return "AttentionRecursiveToken"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["entropy"]
        attentions = stats["attention"]
        ue = []
        for attn, entropy in zip(attentions, entropies):
            entropy = np.array(entropy)
            assert len(attn) == len(entropy)
            attn[1:] /= attn[1:].sum(1)[:, None]
            u = [entropy[0]]
            for i in range(1, len(entropy)):
                u.append(0.5 * entropy[i] + 0.5 * (attn[i - 1, :i] * np.array(u)).sum())
            ue.append(u[:-1])
        return np.concatenate(ue)
