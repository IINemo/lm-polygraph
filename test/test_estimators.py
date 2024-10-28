import random
import numpy as np
import torch
import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.utils.manager import estimate_uncertainty
from lm_polygraph.estimators import *
from lm_polygraph.utils.model import WhiteboxModel

INPUT = "When was Julius Caesar born?"


@pytest.fixture(scope="module")
def model():
    model_path = "bigscience/bloomz-560m"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return WhiteboxModel(base_model, tokenizer)


@pytest.fixture(autouse=True)
def set_random_seeds():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_maximum_sequence_probability(model):
    estimator = MaximumSequenceProbability()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 7.38


def test_perplexity(model):
    estimator = Perplexity()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 2.46


def test_mean_token_entropy(model):
    estimator = MeanTokenEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 4.84


def test_mean_pointwise_mutual_information(model):
    estimator = MeanPointwiseMutualInformation()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -3.94


def test_mean_conditional_pointwise_mutual_information(model):
    estimator = MeanConditionalPointwiseMutualInformation()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -20.57


def test_claim_conditioned_probability(model):
    estimator = ClaimConditionedProbability()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -0.02


def test_ptrue(model):
    estimator = PTrue()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 13.19


def test_ptrue_sampling(model):
    estimator = PTrueSampling()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 12.83


def test_monte_carlo_sequence_entropy(model):
    estimator = MonteCarloSequenceEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 7.21


def test_monte_carlo_normalized_sequence_entropy(model):
    estimator = MonteCarloNormalizedSequenceEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 3.15


def test_lexical_similarity_rouge1(model):
    estimator = LexicalSimilarity(metric="rouge1")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -0.23


def test_lexical_similarity_rouge2(model):
    estimator = LexicalSimilarity(metric="rouge2")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -0.01


def test_lexical_similarity_rougel(model):
    estimator = LexicalSimilarity(metric="rougeL")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -0.23


def test_lexical_similarity_bleu(model):
    estimator = LexicalSimilarity(metric="BLEU")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -0.01


def test_num_sem_sets(model):
    estimator = NumSemSets()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 9


def test_eigval_laplacian_nli_entail(model):
    estimator = EigValLaplacian(similarity_score="NLI_score", affinity="entail")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 7.24


def test_eigval_laplacian_nli_contra(model):
    estimator = EigValLaplacian(similarity_score="NLI_score", affinity="contra")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 4.19


def test_eigval_laplacian_jaccard(model):
    estimator = EigValLaplacian(similarity_score="Jaccard_score")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 4.99


def test_degmat_nli_entail(model):
    estimator = DegMat(similarity_score="NLI_score", affinity="entail")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 0.86


def test_degmat_nli_contra(model):
    estimator = DegMat(similarity_score="NLI_score", affinity="contra")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 0.74


def test_degmat_jaccard(model):
    estimator = DegMat(similarity_score="Jaccard_score")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 0.76


def test_eccentricity_nli_entail(model):
    estimator = Eccentricity(similarity_score="NLI_score", affinity="entail")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 2.83


def test_eccentricity_nli_contra(model):
    estimator = Eccentricity(similarity_score="NLI_score", affinity="contra")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 2.65


def test_eccentricity_jaccard(model):
    estimator = Eccentricity(similarity_score="Jaccard_score")
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 2.83


def test_semantic_entropy(model):
    estimator = SemanticEntropy()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 7.05


def test_sar(model):
    estimator = SAR()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -3.92


def test_token_sar(model):
    estimator = TokenSAR()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 2.64


def test_sentence_sar(model):
    estimator = SentenceSAR()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -1.61


def test_renyi_neg(model):
    estimator = RenyiNeg()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == -22.62


def test_fisher_rao(model):
    estimator = FisherRao()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert round(ue.uncertainty, 2) == 0.79
