import numpy as np
import re

from abc import ABC, abstractmethod
from typing import List

class TokenClusterer(ABC):
    def __init__(self, reduction: str = 'mean'):
        self._reduction = reduction

    def __call__(self, texts: List[str], tokens: List[List[str]], ue: List[float]) -> List[float]:
        assert isinstance(texts, list)
        ue_clustered: List[float] = []
        last = 0
        for text, token in zip(texts, tokens):
            ue_clustered += self._call_single(text, token[:-1], ue[last:last + len(token) - 1])
            last += len(token) - 1
        return ue_clustered

    def _reduce(self, x: List[float]) -> float:
        if self._reduction == 'mean':
            return np.mean(x)
        if self._reduction == 'min':
            return np.min(x)
        if self._reduction == 'max':
            return np.max(x)
        raise Exception(f'Unknown reduction type: {self._reduction}')

    def _call_single(self, text: str, tokens: List[str], ue: List[float]) -> List[float]:
        clusters = np.array(self.cluster(text, tokens))
        ue = np.array(ue)
        new_ue = np.full(ue.shape, fill_value=None)
        for c in np.unique([c for c in clusters if c is not None]):
            new_ue[clusters == c] = self._reduce(ue[clusters == c])
        return new_ue.tolist()

    @abstractmethod
    def cluster(self, text: str, tokens: List[str]) -> List[int]:
        raise Exception('Not implemented')


class IdClusterer(TokenClusterer):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def cluster(self, text: str, tokens: List[str]) -> List[int]:
        return [i for i in range(len(tokens))]
