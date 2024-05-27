import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class PointwiseMutualInformationClaim(Estimator):
    """
    Estimates the claim-level uncertainty of a language model using Pointwise Mutual Information.
    The sequence-level estimation is calculated as product of token-level PMI estimations.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self):
        super().__init__(
            ["greedy_log_likelihoods", "greedy_lm_log_likelihoods", "claims"], "claim"
        )

    def __str__(self):
        return "PointwiseMutualInformationClaim"

    def _reduce(self, x: np.ndarray):
        return -np.sum(x)

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Estimates the mean PMI uncertainties with minus sign for each sample in the input statistics.

        Parameters:
            stats (Dict[str, object]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods',
                * log p(y_i | y_<i) in 'greedy_lm_log_likelihoods',
                * list of extracted claims of type lm_polygraph.stat_calculators.extract_claims.Claim
                  in 'claims'.
        Returns:
            List[List[float]]: float uncertainty for each claim in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["greedy_log_likelihoods"]
        lm_logprobs = stats["greedy_lm_log_likelihoods"]
        claims = stats["claims"]
        claim_ue = []
        for sample_lp, sample_lm_lp, sample_claims in zip(
            logprobs,
            lm_logprobs,
            claims,
        ):
            claim_ue.append([])
            for claim in sample_claims:
                sample_lm_lp[0] = 0
                mi_scores = np.array(sample_lp) - np.array(sample_lm_lp)
                tokens = np.array(claim.aligned_token_ids)
                claim_pmi = mi_scores[tokens]
                claim_ue[-1].append(self._reduce(claim_pmi))
        return claim_ue
