import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class RandomBaselineClaim(Estimator):
    """
    Estimates the claim-level maximum token entropy.
    """

    def __init__(self):
        super().__init__(["claims"], "claim")

    def __str__(self):
        return "RandomBaselineClaim"

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Random baseline

        Parameters:
            stats (Dict[str, object]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * list of extracted claims of type lm_polygraph.stat_calculators.extract_claims.Claim
                  in 'claims'.
        Returns:
            List[List[float]]: concatenated minus log probabilities for each claim.
                Higher values indicate more uncertain samples.
        """
        claims = stats["claims"]
        claim_ue = []
        for sample_claims in claims:
            claim_ue.append([])
            for _ in sample_claims:
                claim_ue[-1].append(np.random.rand())
        return claim_ue
