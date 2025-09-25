import os
import torch
import pytest

from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import *
from lm_polygraph.model_adapters.blackbox_model import BlackboxModel

INPUT = "When was Julius Caesar born?"


@pytest.fixture(scope="module")
def model():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    return BlackboxModel('gpt-4.1-nano')


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


def test_sentence_sar(model):
    estimator = SentenceSAR()
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


def test_semantic_density_concat(model):
    estimator = SemanticDensity()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_semantic_density(model):
    estimator = SemanticDensity(concat_input=False)
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)
