import numpy as np
import torch
from typing import Dict
from .estimator import Estimator
from boostedprob import calculate_boostedprob

import numpy as np

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class BoostedProbClaim(Estimator):
    """
    Estimates the claim-level uncertainty of a language model using boosted model probability, i.e.,
    - When the output token has dominant probability mass: returns the sum of probabilities of all dominant tokens
    - When the token is not dominant: returns the probability of the token itself
    Details can be found in the paper: https://aclanthology.org/2025.emnlp-main.166.pdf
    """
    def __init__(self):
        super().__init__(["greedy_log_probs", "claims"], "claim")

    def __str__(self):
        return "BoostedProbClaim"

    def _reduce(self, x: np.ndarray):
        return -np.mean(x)

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Estimates token-level BoostedProb

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * Full distribution of log p(y_i | y_<i, x) in 'greedy_log_probs'
        Returns:
            np.ndarray: claim-level BoostedProb. Higher values indicate more uncertainty.
        """
        lprob_distributions = stats["greedy_log_probs"]
        output_tokens = stats["greedy_tokens"]
        claims = stats["claims"]
        
        boostedprobs = [calculate_boostedprob(torch.tensor(lprob_distribution), torch.tensor(output_tokens)) 
                 for lprob_distribution in lprob_distributions]
        claim_ue = []
        for sample_boostedprob, sample_claims in zip(boostedprobs, claims):
            claim_ue.append([])
            for claim in sample_claims:
                tokens = np.array(claim.aligned_token_ids)
                claim_probs = np.array(sample_boostedprob)[tokens]
                claim_ue[-1].append(self._reduce(claim_probs))
        return claim_ue
