import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class PerplexityClaim(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods", "claims"], "claim")

    def __str__(self):
        return "PerplexityClaim"

    def _reduce(self, x: np.ndarray):
        return -np.mean(x)

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        log_likelihoods = stats["greedy_log_likelihoods"]
        claims = stats["claims"]
        claim_ue = []
        for sample_ll, sample_claims in zip(log_likelihoods, claims):
            claim_ue.append([])
            for claim in sample_claims:
                tokens = np.array(claim.aligned_token_ids)
                claim_ll = np.array(sample_ll)[tokens]
                claim_ue[-1].append(self._reduce(claim_ll))
        return claim_ue
