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
            if not np.issubdtype(tokens.dtype, np.integer):
                print("⚠️ Warning: Non-integer tokens found, skipping claim.")
                continue

            if tokens.size == 0 or np.any(tokens >= len(sample_ll)):
                print(f"⚠️ Skipping due to empty or out-of-bound tokens: {tokens}")
                continue 
                claim_ll = np.array(sample_ll)[tokens]
                claim_ue[-1].append(self._reduce(claim_ll))
        return claim_ue
