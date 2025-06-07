import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.token_sar import token_level_sar_scores


class TokenSARClaim(Estimator):
    """
    Estimator for computing TokenSAR scores for claims.

    This estimator adapts the TokenSAR method to the claim level. It aggregates
    token-level semantic alignment scores (E_t) for each claim by summing the
    scores of tokens aligned to that claim.

    TokenSAR is a metric for semantic agreement between generated tokens and
    source context, factoring in both likelihood and semantic similarity.

    Required statistics:
        - "token_similarity": Token-level similarity between generated tokens and context
        - "greedy_log_likelihoods": Log-likelihoods of greedy-decoded tokens
        - "claims": List of Claim objects with aligned_token_ids

    Provides:
        - List[List[float]]: TokenSAR scores for each claim per sample
    """

    def __init__(self):
        super().__init__(["token_similarity", "greedy_log_likelihoods"], "claim")

    def __str__(self):
        return "TokenSARClaim"

    def __call__(self, stats: Dict[str, np.ndarray]) -> List[List[float]]:
        """
        Calculates TokenSAR scores for all claims based on aligned token scores.

        Args:
            stats (Dict): Dictionary containing:
                - "token_similarity": np.ndarray with similarity scores
                - "greedy_log_likelihoods": np.ndarray of token log-likelihoods
                - "claims": List[List[Claim]] with aligned token indices

        Returns:
            List[List[float]]: TokenSAR scores per claim, structured as:
                [n_samples, n_claims_in_sample]
        """
        all_E_t = token_level_sar_scores(stats)
        token_sar: List[List[float]] = []
        for E_t, claims in zip(all_E_t, stats["claims"]):
            token_sar.append([])
            for claim in claims:
                aligned_token_ids = np.array(claim.aligned_token_ids)
                token_sar[-1].append(E_t[aligned_token_ids].sum())
        return token_sar
