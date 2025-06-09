"""
Uncertainty estimation methods for language models.

This module provides a comprehensive collection of uncertainty estimation methods
for both white-box models (with access to logits) and black-box models (text-only).
Each estimator inherits from the base Estimator class and implements a specific
approach to quantifying model uncertainty.

Categories of Estimators:

Information-Based (White-box only):
    - Perplexity: Average negative log-likelihood
    - TokenEntropy/MeanTokenEntropy: Shannon entropy of token distributions
    - MaximumSequenceProbability: Joint probability of generated sequence
    - PointwiseMutualInformation: Information-theoretic measure
    - ConditionalPointwiseMutualInformation: Conditional variant of PMI

Meaning Diversity (Black-box compatible):
    - SemanticEntropy: Entropy over semantic equivalence classes
    - LexicalSimilarity: ROUGE-based similarity between samples
    - NumSemSets: Number of distinct semantic clusters
    - EigValLaplacian: Eigenvalue analysis of semantic graph
    - SAR/TokenSAR/SentenceSAR: Self-alignment rate metrics

Density-Based (White-box, requires training):
    - MahalanobisDistance: Distance in feature space
    - RelativeMahalanobisDistance: Normalized Mahalanobis distance
    - RDE: Robust density estimation
    - PPLMDSeq: Combined perplexity and Mahalanobis

Reflexive (Model self-evaluation):
    - PTrue: Probability of "True" when asked about correctness
    - PTrueSampling: PTrue with multiple samples
    - Verbalized1S/2S: Verbal confidence expressions
    - Linguistic1S: Linguistic uncertainty cues

Ensemble-Based (Requires multiple models):
    - Various EPT/PET methods for token and sequence level

Usage Example:
    >>> from lm_polygraph import WhiteboxModel
    >>> from lm_polygraph.estimators import TokenEntropy
    >>> from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
    >>> 
    >>> model = WhiteboxModel.from_pretrained("gpt2")
    >>> estimator = TokenEntropy()
    >>> result = estimate_uncertainty(model, estimator, "What is AI?")
    >>> print(f"Uncertainty: {result.uncertainty}")

See the documentation for each estimator class for detailed information
about requirements, parameters, and appropriate use cases.
"""

from .estimator import Estimator
from .claim.claim_conditioned_probability import ClaimConditionedProbabilityClaim
from .claim.max_probability import MaximumClaimProbability
from .claim.p_true import PTrueClaim
from .claim.attention_score import AttentionScoreClaim
from .claim.perplexity import PerplexityClaim
from .claim.token_entropy import MaxTokenEntropyClaim
from .claim.pointwise_mutual_information import PointwiseMutualInformationClaim
from .claim.focus import FocusClaim
from .claim.frequency_scoring import FrequencyScoringClaim
from .claim.token_sar import TokenSARClaim
from .max_probability import (
    MaximumSequenceProbability,
    MaximumTokenProbability,
)
from .attention_score import AttentionScore
from .claim_conditioned_probability import ClaimConditionedProbability
from .token_entropy import MeanTokenEntropy, TokenEntropy
from .pointwise_mutual_information import (
    MeanPointwiseMutualInformation,
    PointwiseMutualInformation,
)
from .conditional_pointwise_mutual_information import (
    MeanConditionalPointwiseMutualInformation,
    ConditionalPointwiseMutualInformation,
)
from .p_true import PTrue
from .p_true_sampling import PTrueSampling
from .monte_carlo_sequence_entropy import MonteCarloSequenceEntropy
from .monte_carlo_normalized_sequence_entropy import MonteCarloNormalizedSequenceEntropy
from .lexical_similarity import LexicalSimilarity
from .deg_mat import DegMat
from .eccentricity import Eccentricity
from .eig_val_laplacian import EigValLaplacian
from .num_sem_sets import NumSemSets
from .semantic_entropy import SemanticEntropy
from .semantic_entropy_token import SemanticEntropyToken
from .perplexity import Perplexity
from .mahalanobis_distance import MahalanobisDistanceSeq
from .relative_mahalanobis_distance import RelativeMahalanobisDistanceSeq
from .rde import RDESeq
from .ppl_md import PPLMDSeq
from .ensemble_token_measures import (
    EPTtu,
    EPTdu,
    EPTmi,
    EPTrmi,
    EPTepkl,
    EPTent5,
    EPTent10,
    EPTent15,
    PETtu,
    PETdu,
    PETmi,
    PETrmi,
    PETepkl,
    PETent5,
    PETent10,
    PETent15,
)
from .ensemble_sequence_measures import (
    EPStu,
    EPSrmi,
    EPSrmiabs,
    PEStu,
    PESrmi,
    PESrmiabs,
)
from .token_sar import TokenSAR
from .sentence_sar import SentenceSAR
from .sar import SAR
from .renyi_neg import RenyiNeg
from .fisher_rao import FisherRao
from .verbalized_1s import Verbalized1S
from .verbalized_2s import Verbalized2S
from .linguistic_1s import Linguistic1S
from .claim.random_baseline import RandomBaselineClaim
from .label_prob import LabelProb
from .p_true_empirical import PTrueEmpirical

from .focus import Focus
from .kernel_language_entropy import KernelLanguageEntropy
from .luq import LUQ
from .eigenscore import EigenScore
