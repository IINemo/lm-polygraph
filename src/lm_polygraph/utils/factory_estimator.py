from importlib import import_module
from lm_polygraph.estimators import *


def load_simple_estimators(name, config):
    SIMPLE_ESTIMATORS = [
        MaximumSequenceProbability,
        Perplexity,
        MeanTokenEntropy,
        MeanPointwiseMutualInformation,
        MeanConditionalPointwiseMutualInformation,
        ClaimConditionedProbability,
        PTrue,
        PTrueSampling,
        MonteCarloSequenceEntropy,
        MonteCarloNormalizedSequenceEntropy,
        NumSemSets,
        SemanticEntropy,
        SAR,
        TokenSAR,
        SentenceSAR,
        RenyiNeg,
        FisherRao,
        MahalanobisDistanceSeq,
        RelativeMahalanobisDistanceSeq,
        RDESeq,
        PPLMDSeq,
        MaximumClaimProbability,
        PerplexityClaim,
        MaxTokenEntropyClaim,
        PointwiseMutualInformationClaim,
        PTrueClaim,
        ClaimConditionedProbabilityClaim,
        RandomBaselineClaim,
    ]

    try:
        simple_estimators = {e.__name__: e for e in SIMPLE_ESTIMATORS}
        est = simple_estimators[name](**config)
        return est

    except KeyError:
        return None


class FactoryEstimator:
    def __call__(self, name, config):
        est = load_simple_estimators(name, config)
        if est is not None:
            return est

        module = import_module(name)
        return module.load_estimator(config)
