import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class MaxTokenEntropyClaim(Estimator):
    """
    Estimates the claim-level maximum token entropy.
    """

    def __init__(self):
        super().__init__(["entropy", "claims"], "claim")

    def __str__(self):
        return "MaxTokenEntropyClaim"

    def _reduce(self, x: np.ndarray):
        return np.max(x)

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Estimates the minus log-probability of each claim in input statistics.

        Parameters:
            stats (Dict[str, object]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * list of extracted claims of type lm_polygraph.stat_calculators.extract_claims.Claim
                  in 'claims'.
        Returns:
            List[List[float]]: concatenated minus log probabilities for each claim.
                Higher values indicate more uncertain samples.
        """
        entropies = stats["entropy"]
        claims = stats["claims"]
        claim_ue = []
        for sample_ent, sample_claims in zip(entropies, claims):
            claim_ue.append([])
            for claim in sample_claims:
                tokens = np.array(claim.aligned_token_ids)
                claim_ent = np.array(sample_ent)[tokens]
                claim_ue[-1].append(self._reduce(claim_ent))
        return claim_ue
