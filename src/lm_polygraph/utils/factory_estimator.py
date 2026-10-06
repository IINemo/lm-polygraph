from importlib import import_module
import lm_polygraph.estimators as estimators
from lm_polygraph.estimators import Estimator


def load_simple_estimators(name: str, config):
    names = [
        "MaximumSequenceProbability",
        "Perplexity",
        "MeanTokenEntropy",
        "SelfCertainty",
        "MeanPointwiseMutualInformation",
        "MeanConditionalPointwiseMutualInformation",
        "ClaimConditionedProbability",
        "PTrue",
        "PTrueSampling",
        "MonteCarloSequenceEntropy",
        "MonteCarloNormalizedSequenceEntropy",
        "NumSemSets",
        "LexicalSimilarity",
        "EigValLaplacian",
        "DegMat",
        "Eccentricity",
        "SemanticEntropy",
        "SAR",
        "TokenSAR",
        "SentenceSAR",
        "LUQ",
        "KernelLanguageEntropy",
        "EigenScore",
        "RenyiNeg",
        "FisherRao",
        "MahalanobisDistanceSeq",
        "RelativeMahalanobisDistanceSeq",
        "RDESeq",
        "PPLMDSeq",
        "LabelProb",
        "PTrueEmpirical",
        "Verbalized1S",
        "Verbalized2S",
        "Linguistic1S",
        "MaximumClaimProbability",
        "Focus",
        "PerplexityClaim",
        "MaxTokenEntropyClaim",
        "SelfCertaintyClaim",
        "PointwiseMutualInformationClaim",
        "PTrueClaim",
        "ClaimConditionedProbabilityClaim",
        "RandomBaselineClaim",
        "FrequencyScoringClaim",
        "TokenSARClaim",
        "FocusClaim",
        "AttentionScore",
        "AttentionScoreClaim",
        "BoostedProbClaim",
        "CocoaMSP",
        "CocoaPPL",
        "CocoaMTE",
        "RAUQ",
        "CSL",
        "SemanticDensity",
        "BoostedProbSequence",
        "PredictiveKernelEntropy",
        "SpectralUncertainty",
    ]
    if name not in names:
        return None
    return getattr(estimators, name)(**config)


class FactoryEstimator:
    """Constructs an estimator from a given name and configuration."""

    def __call__(self, name: str, config) -> Estimator:
        est = load_simple_estimators(name, config)
        if est is not None:
            return est

        module = import_module(name)
        return module.load_estimator(config)
