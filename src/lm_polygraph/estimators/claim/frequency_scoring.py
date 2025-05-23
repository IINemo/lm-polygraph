from typing import Dict, List
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.stat_calculators.extract_claims import Claim


class FrequencyScoringClaim(Estimator):
    """
    Estimator for calculating a simple frequency-based score for claims.

    This baseline scores each claim based on the number of NLI "contradiction"
    labels minus the number of "entailment" labels across all text samples.

    This scoring strategy is introduced in https://arxiv.org/abs/2402.10978

    Unlike the original paper, which uses GPT for entailment/contradiction judgments,
    this implementation uses outputs from a separate NLI model to compute the scores.

    For each claim:
        Score = Count("contra") - Count("entail")

    This scoring does not account for NLI probability magnitudes or contextual
    richness, serving as a basic contrastive count metric.

    Required statistics:
        - "claims": List of Claim objects for each sample
        - "nli_to_sample_texts": NLI labels between each claim and associated contexts
    """

    def __init__(self):
        """
        Initializes the FrequencyScoringClaim estimator.
        """
        dependencies = ["claims", "nli_to_sample_texts"]
        super().__init__(dependencies, "claim")

    def __str__(self):
        return "FrequencyScoringClaim"

    def __call__(self, stats: Dict[str, object]) -> List[List[float]]:
        """
        Calculates frequency-based scores for each claim.

        Args:
            stats (Dict): Dictionary with keys:
                - "claims": List[List[Claim]]
                - "nli_to_sample_texts": List[List[List[Dict[str, float]]]], where each inner list
                  contains NLI labels ("entail", "neutral", "contra") between a claim
                  and sample texts.

        Returns:
            List[List[float]]: Frequency-based scores per claim, structured as:
                [n_samples, n_claims_in_sample]
        """
        claims: List[List[Claim]] = stats["claims"]
        nli_to_sample_texts: List[List[List[Dict[str, float]]]] = stats[
            "nli_to_sample_texts"
        ]
        scores = []

        for sample_claims, sample_nli_labels in zip(claims, nli_to_sample_texts):
            sample_scores = []
            for claim_nlis in sample_nli_labels:
                hard_classes: List[str] = [max(c, key=c.get) for c in claim_nlis]
                contra_count = hard_classes.count("contra")
                entail_count = hard_classes.count("entail")
                score = float(contra_count - entail_count)
                sample_scores.append(score)
            scores.append(sample_scores)

        return scores
