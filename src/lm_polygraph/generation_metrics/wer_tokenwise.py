import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class WERTokenwiseMetric(GenerationMetric):
    def __init__(self):
        super().__init__(["greedy_tokens"], "token")

    def __str__(self):
        return f"WERTokenwise"

    def _score_single(self, hyp: List[int], ref: List[int]) -> List[float]:
        n, m = len(hyp), len(ref)
        dp = np.zeros(shape=(n + 1, m + 1))
        for i in range(n + 1):
            dp[i, 0] = i
        for i in range(m + 1):
            dp[0, i] = i
        for i in range(len(hyp)):
            for j in range(len(ref)):
                if hyp[i] == ref[j]:
                    dp[i + 1, j + 1] = dp[i, j]
                else:
                    dp[i + 1, j + 1] = min(dp[i + 1, j], dp[i, j + 1], dp[i, j]) + 1
        i, j = n, m
        correct: List[float] = [0 for _ in range(n)]
        while i != 0 or j != 0:
            if i == 0:
                break
            if j == 0:
                break
            if hyp[i - 1] == ref[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
                correct[i - 1] = 1
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i, j - 1] + 1:
                j -= 1
            elif dp[i, j] == dp[i - 1, j] + 1:
                i -= 1
            elif dp[i, j] == dp[i - 1, j - 1] + 1:
                i -= 1
                j -= 1
            else:
                raise Exception("Internal error")

        return correct[:-1]

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        target_tokens: List[List[int]],
    ) -> np.ndarray:
        return np.array(
            [
                c
                for hyp, ref in zip(stats["greedy_tokens"], target_tokens)
                for c in self._score_single(hyp, ref)
            ]
        )
