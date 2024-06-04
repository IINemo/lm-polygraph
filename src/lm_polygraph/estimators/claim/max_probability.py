import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class MaximumClaimProbability(Estimator):
    """
    Estimates the claim-level uncertainty of a language model by calculating the
    log-probability for each claim during autoregressive generation.
    """

    def __init__(self):
        super().__init__(["greedy_log_likelihoods", "claims"], "claim")

    def __str__(self):
        return "MaximumClaimProbability"

    def _reduce(self, x: np.ndarray):
        return -np.sum(x)

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Estimates the minus log-probability of each claim in input statistics.

        Parameters:
            stats (Dict[str, object]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
                * list of extracted claims of type lm_polygraph.stat_calculators.extract_claims.Claim
                  in 'claims'
        Returns:
            List[List[float]]: concatenated minus log probabilities for each claim.
                Higher values indicate more uncertain samples.
        """
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
