import torch
import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import *
from lm_polygraph.utils.model import WhiteboxModel

INPUT = "When was Julius Caesar born?"


@pytest.fixture(scope="module")
def model():
    model_path = "bigscience/bloomz-560m"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return WhiteboxModel(base_model, tokenizer)


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


def test_focus(model):
    model_name = model.model.config._name_or_path
    estimator = Focus(
        model_name=model_name,
        path="../token_idf/{model_name}/token_idf.pkl",
        gamma=0.9,
        p=0.01,
        idf_dataset="LM-Polygraph/RedPajama-Data-100-Sample-For-Test",
        trust_remote_code=True,
        idf_seed=42,
        idf_dataset_size=5,
        spacy_path="en_core_web_sm",
    )
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


def test_attentionscore(model):
    estimator = AttentionScore()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_cocoamsp(model):
    estimator = CocoaMSP()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_cocoappl(model):
    estimator = CocoaPPL()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_cocoamte(model):
    estimator = CocoaMTE()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_rauq(model):
    estimator = RAUQ()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_csl(model):
    estimator = CSL()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_semantic_density_concat(model):
    estimator = SemanticDensity()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_semantic_density(model):
    estimator = SemanticDensity(concat_input=False)
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)
