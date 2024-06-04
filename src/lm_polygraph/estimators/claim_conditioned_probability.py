import numpy as np

from typing import Dict

from .estimator import Estimator


class ClaimConditionedProbability(Estimator):
    def __init__(self):
        super().__init__(
            [
                "greedy_tokens",
                "greedy_tokens_alternatives",
                "greedy_tokens_alternatives_nli",
            ],
            "sequence",
        )

    def __str__(self):
        return "CCP"

    def _reduce(self, logprobs: list[float]):
        return np.exp(np.sum(logprobs))

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        words = stats["greedy_tokens"]
        alternatives = stats["greedy_tokens_alternatives"]
        alternatives_nli = stats["greedy_tokens_alternatives_nli"]
        prob_nli = []
        for sample_words, sample_alternatives, sample_alternatives_nli in zip(
            words,
            alternatives,
            alternatives_nli,
        ):
            sample_mnlis = []
            for word, word_alternatives, word_alternatives_nli in zip(
                sample_words,
                sample_alternatives,
                sample_alternatives_nli,
            ):
                entail_logprobs, entail_words = [], []
                contra_logprobs, contra_words = [], []
                for i in range(len(word_alternatives)):
                    word_alt, logprob = word_alternatives[i]
                    if i == 0 or word_alternatives_nli[0][i] == "entail":
                        entail_logprobs.append(logprob)
                        entail_words.append(word_alt)
                    elif word_alternatives_nli[0][i] == "contra":
                        contra_logprobs.append(logprob)
                        contra_words.append(word_alt)
                entail_logprob = np.logaddexp.reduce(entail_logprobs)
                total_logprob = np.logaddexp.reduce(entail_logprobs + contra_logprobs)
                sample_mnlis.append(entail_logprob - total_logprob)
            prob_nli.append(self._reduce(sample_mnlis))
        return -np.array(prob_nli)
