import numpy as np

from typing import Dict

from .estimator import Estimator


class AttentionEntropySeq(Estimator):
    def __init__(self):
        super().__init__(["entropy", "attention"], "sequence")

    def __str__(self):
        return "AttentionEntropySeq"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["entropy"]
        attentions = stats["attention"]
        ue = []
        for attn, entropy in zip(attentions, entropies):
            entropy = np.array(entropy)
            attn[1:] /= attn[1:].sum(1)[:, None]
            np.fill_diagonal(attn, 1)
            attn /= attn.sum(1)[:, None]
            ue.append(np.mean(attn.sum(0) * entropy))
        return np.array(ue)


class AttentionEntropyToken(Estimator):
    def __init__(self):
        super().__init__(["entropy", "attention"], "token")

    def __str__(self):
        return "AttentionEntropyToken"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["entropy"]
        attentions = stats["attention"]
        ue = []
        for attn, entropy in zip(attentions, entropies):
            entropy = np.array(entropy)
            attn[1:] /= attn[1:].sum(1)[:, None]
            np.fill_diagonal(attn, 1)
            attn /= attn.sum(1)[:, None]
            ue.append((attn.sum(0) * entropy)[:-1])
        return np.concatenate(ue)
