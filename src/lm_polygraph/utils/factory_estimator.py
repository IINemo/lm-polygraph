from importlib import import_module
from lm_polygraph.estimators import *


def load_simple_estimators(name: str, config):
    SIMPLE_ESTIMATORS = [
        MaximumSequenceProbability,
        CumulativeMaxSequenceProbability,
        SortedCumulativeMaxSequenceProbability,
        Perplexity,
        CumulativePerplexity,
        SortedCumulativePerplexity,
        MeanTokenEntropy,
        CumulativeMeanTokenEntropy,
        SortedCumulativeMeanTokenEntropy,
        MeanPointwiseMutualInformation,
        MeanConditionalPointwiseMutualInformation,
        ClaimConditionedProbability,
        PTrue,
        PTrueSampling,
        MonteCarloSequenceEntropy,
        MonteCarloNormalizedSequenceEntropy,
        NumSemSets,
        LexicalSimilarity,
        EigValLaplacian,
        DegMat,
        Eccentricity,
        SemanticEntropy,
        SAR,
        TokenSAR,
        SentenceSAR,
        LUQ,
        KernelLanguageEntropy,
        EigenScore,
        RenyiNeg,
        FisherRao,
        MahalanobisDistanceSeq,
        RelativeMahalanobisDistanceSeq,
        RDESeq,
        PPLMDSeq,
        LabelProb,
        PTrueEmpirical,
        Verbalized1S,
        Verbalized2S,
        Linguistic1S,
        MaximumClaimProbability,
        Focus,
        PerplexityClaim,
        MaxTokenEntropyClaim,
        PointwiseMutualInformationClaim,
        PTrueClaim,
        ClaimConditionedProbabilityClaim,
        RandomBaselineClaim,
        FrequencyScoringClaim,
        TokenSARClaim,
        FocusClaim,
        AttentionScore,
        AttentionScoreClaim,
    ]

    simple_estimators = {e.__name__: e for e in SIMPLE_ESTIMATORS}
    try:
        est_class = simple_estimators[name]
    except KeyError:
        return None

    return est_class(**config)


class FactoryEstimator:
    """Constructs an estimator from a given name and configuration."""

    def __call__(self, name: str, config) -> Estimator:
        est = load_simple_estimators(name, config)
        if est is not None:
            return est

        module = import_module(name)
        return module.load_estimator(config)
