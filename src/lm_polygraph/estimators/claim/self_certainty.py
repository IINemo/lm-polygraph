import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


class SelfCertaintyClaim(Estimator):
    """
    Computes a self-certainty metric for each claim within a language model's output.
    The metric estimates the KL divergence between a uniform distribution and the model’s
    predicted token distribution at each token position. This is done by averaging
    the per-token divergences over the tokens aligned with each claim.

    A higher self-certainty score indicates higher uncertainty for the given claim.

    Reference:
        "Scalable Best-of-N Selection for Large Language Models via Self-Certainty"
        https://arxiv.org/pdf/2502.18581
    """

    def __init__(self):
        super().__init__(["greedy_log_probs", "claims"], "claim")

    def __str__(self):
        return "SelfCertaintyClaim"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes self-certainty scores for each claim in the batch. For each token,
        the score is based on the KL divergence from a uniform distribution
        to the predicted distribution, approximated as:

            KL(uniform || predicted) ≈ -mean(log p) - log(V)

        The claim-level score is computed by averaging the token-level
        self-certainties over the tokens aligned with each claim.

        Parameters:
            stats (Dict[str, np.ndarray]):
                - 'greedy_log_probs': list of list of np.ndarrays. Each inner array contains
                  log-probabilities over the vocabulary for a token.
                - 'claims': list of claim lists. Each claim object must have
                  an 'aligned_token_ids' attribute indicating the tokens covered by the claim.

        Returns:
            np.ndarray: A nested list (or array) of self-certainty scores, one per claim per sample.
                        Higher values mean higher uncertainty for that claim.
        """
        logprobs = stats["greedy_log_probs"]
        claims = stats["claims"]
        self_certainties: list[list[float]] = []

        for s_lp, sample_claims in zip(logprobs, claims):  # iterate over samples
            token_self_certainties = []
            for lp in s_lp:  # iterate over tokens
                token_logprobs = np.array(lp[~np.isinf(lp)])  # filter out -inf values
                # Estimate KL(uniform || predicted)
                token_self_certainties.append(
                    -np.mean(token_logprobs) - np.log(len(token_logprobs))
                )
            token_self_certainties = np.array(token_self_certainties)

            # Compute claim-level certainty by averaging over aligned tokens
            sample_claim_scores = []
            for claim in sample_claims:
                tokens = np.array(claim.aligned_token_ids)
                claim_sc = -np.mean(token_self_certainties[tokens])
                sample_claim_scores.append(claim_sc)

            self_certainties.append(sample_claim_scores)

        return self_certainties
