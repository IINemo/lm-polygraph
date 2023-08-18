import numpy as np

from typing import Dict

from .estimator import Estimator


class ConditionalMutualInformationSeq(Estimator):
    def __init__(self, tau: float, lambd: float):
        super().__init__(['greedy_log_likelihoods', 'greedy_lm_log_likelihoods', 'entropy'], 'sequence')
        self.tau = tau
        self.lambd = lambd

    def __str__(self):
        return 'ConditionalMutualInformationSeq'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats['greedy_log_likelihoods']
        lm_logprobs = stats['greedy_lm_log_likelihoods']
        entropies = stats['entropy']
        mi_scores = []
        for lp, lm_lp, ent in zip(logprobs, lm_logprobs, entropies):
            mi_scores.append([])
            for t in range(len(lp)):
                score = lp[t]
                if t > 0 and ent[t] >= self.tau:
                    score -= self.lambd * lm_lp[t]
                mi_scores[-1].append(score)
        return np.array([-np.mean(sc) for sc in mi_scores])


class ConditionalMutualInformationToken(Estimator):
    def __init__(self, tau: float, lambd: float):
        super().__init__(['greedy_log_likelihoods', 'greedy_lm_log_likelihoods', 'entropy'], 'token')
        self.tau = tau
        self.lambd = lambd

    def __str__(self):
        return 'ConditionalMutualInformationToken'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        logprobs = stats['greedy_log_likelihoods']
        lm_logprobs = stats['greedy_lm_log_likelihoods']
        entropies = stats['entropy']
        mi_scores = []
        for lp, lm_lp, ent in zip(logprobs, lm_logprobs, entropies):
            mi_scores.append([])
            for t in range(len(lp)):
                score = lp[t]
                if t > 0 and ent[t] >= self.tau:
                    score -= self.lambd * lm_lp[t]
                mi_scores[-1].append(score)
        return np.concatenate([-np.array(sc[:-1]) for sc in mi_scores])
