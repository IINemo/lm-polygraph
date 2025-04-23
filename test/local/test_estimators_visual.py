import torch
import pytest

from transformers import AutoModelForVision2Seq, AutoProcessor

from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import *
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel

INPUT = "<grounding>An image of?"


@pytest.fixture(scope="module")
def model():
    model_path = "microsoft/kosmos-2-patch14-224"
    image_urls = [
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
    ]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    base_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)

    return VisualWhiteboxModel(base_model, processor, image_urls=image_urls)


def test_maximum_sequence_probability(model):
    estimator = MaximumSequenceProbability()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_perplexity(model):
    estimator = Perplexity()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_mean_token_entropy(model):
    estimator = MeanTokenEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_mean_pointwise_mutual_information(model):
    estimator = MeanPointwiseMutualInformation()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_mean_conditional_pointwise_mutual_information(model):
    estimator = MeanConditionalPointwiseMutualInformation()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_claim_conditioned_probability(model):
    estimator = ClaimConditionedProbability()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_ptrue(model):
    estimator = PTrue()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_ptrue_sampling(model):
    estimator = PTrueSampling()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_monte_carlo_sequence_entropy(model):
    estimator = MonteCarloSequenceEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_monte_carlo_normalized_sequence_entropy(model):
    estimator = MonteCarloNormalizedSequenceEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_lexical_similarity_rouge1(model):
    estimator = LexicalSimilarity(metric="rouge1")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_lexical_similarity_rouge2(model):
    estimator = LexicalSimilarity(metric="rouge2")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_lexical_similarity_rougel(model):
    estimator = LexicalSimilarity(metric="rougeL")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_lexical_similarity_bleu(model):
    estimator = LexicalSimilarity(metric="BLEU")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_num_sem_sets(model):
    estimator = NumSemSets()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eigval_laplacian_nli_entail(model):
    estimator = EigValLaplacian(similarity_score="NLI_score", affinity="entail")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eigval_laplacian_nli_contra(model):
    estimator = EigValLaplacian(similarity_score="NLI_score", affinity="contra")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eigval_laplacian_jaccard(model):
    estimator = EigValLaplacian(similarity_score="Jaccard_score")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_degmat_nli_entail(model):
    estimator = DegMat(similarity_score="NLI_score", affinity="entail")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_degmat_nli_contra(model):
    estimator = DegMat(similarity_score="NLI_score", affinity="contra")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_degmat_jaccard(model):
    estimator = DegMat(similarity_score="Jaccard_score")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eccentricity_nli_entail(model):
    estimator = Eccentricity(similarity_score="NLI_score", affinity="entail")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eccentricity_nli_contra(model):
    estimator = Eccentricity(similarity_score="NLI_score", affinity="contra")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eccentricity_jaccard(model):
    estimator = Eccentricity(similarity_score="Jaccard_score")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_semantic_entropy(model):
    estimator = SemanticEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_sar(model):
    estimator = SAR()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_token_sar(model):
    estimator = TokenSAR()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_sentence_sar(model):
    estimator = SentenceSAR()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_renyi_neg(model):
    estimator = RenyiNeg()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_fisher_rao(model):
    estimator = FisherRao()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_kernel_language_entropy(model):
    estimator = KernelLanguageEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_luq(model):
    estimator = LUQ()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_eigenscore(model):
    estimator = EigenScore()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)
