from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator


class PTrueClaim(Estimator):
    """
    Estimates the claim-level uncertainty of a language model following the method of
    "P(True)" as provided in the paper https://arxiv.org/abs/2207.05221.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    On input question q and model generation a, the method uses the following prompt:
        Question: {q}
        Possible answer: {a}
        Is the possible answer True or False?
        The possible answer is:
    and calculates the log-probability of 'True' token with minus sign.
    """

    def __init__(self):
        super().__init__(["p_true_claim", "claims"], "claim")

    def __str__(self):
        return "PTrueClaim"

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Estimates minus log-probability of True token for each sample in the input statistics.

        Parameters:
            stats (Dict[str, object]): input statistics, which for multiple samples includes:
                * probabilities of the true token in 'p_true'
        Returns:
            List[List[float]]: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        claims = stats["claims"]
        ptrue = stats["p_true_claim"]
        claim_ue = []
        j = 0
        for sample_claims in claims:
            claim_ue.append([])
            for _ in sample_claims:
                claim_ue[-1].append(-ptrue[j])
                j += 1
        return claim_ue
