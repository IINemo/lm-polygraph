import torch
import pytest
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import *
from lm_polygraph.utils.model import WhiteboxModel

INPUT = "When was Julius Caesar born?"

# Test data for BayesPE
TEST_TEXTS = [
    "I love this product! It's amazing.",
    "This is terrible, I hate it.",
    "The product is okay, nothing special.",
    "Absolutely fantastic experience!",
    "Would not recommend to anyone."
]

TEST_LABELS = [1, 0, 0, 1, 0]  # 1 for positive, 0 for negative

FEW_SHOT_EXAMPLES = [
    {"text": "This movie was great!", "label": "positive"},
    {"text": "I didn't enjoy it at all.", "label": "negative"},
    {"text": "The service was excellent.", "label": "positive"},
    {"text": "Waste of money.", "label": "negative"}
]


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


def test_boostedprob_sequence(model):
    estimator = BoostedProbSequence()
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)


def test_bayespe_zero_shot(model):
    estimator = BayesPEZeroShot(
        instructions=[
            "classify the sentiment of the text",
            "determine if the text is positive or negative",
            "what is the emotional tone of the text"
        ],
        n_forward_passes=3
    )
    
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)
    
    estimator.optimize_weights(TEST_TEXTS, TEST_LABELS)
    assert isinstance(estimator.weights, np.ndarray)
    assert len(estimator.weights) == len(estimator.instructions)
    assert np.allclose(np.sum(estimator.weights), 1.0, atol=1e-6)
    
    uncertainties = estimator({"input_texts": TEST_TEXTS})
    assert isinstance(uncertainties, np.ndarray)
    assert len(uncertainties) == len(TEST_TEXTS)
    assert np.all(uncertainties >= 0)  # Uncertainties should be non-negative


def test_bayespe_few_shot(model):
    estimator = BayesPEFewShot(
        instructions=[
            "classify the sentiment of the text",
            "determine if the text is positive or negative",
            "what is the emotional tone of the text"
        ],
        few_shot_examples=FEW_SHOT_EXAMPLES,
        n_forward_passes=3
    )
    
    ue = estimate_uncertainty(model, estimator, INPUT)
    assert isinstance(ue.uncertainty, float)
    
    estimator.optimize_weights(TEST_TEXTS, TEST_LABELS)
    assert isinstance(estimator.weights, np.ndarray)
    assert len(estimator.weights) == len(estimator.instructions)
    assert np.allclose(np.sum(estimator.weights), 1.0, atol=1e-6)
    
    uncertainties = estimator({"input_texts": TEST_TEXTS})
    assert isinstance(uncertainties, np.ndarray)
    assert len(uncertainties) == len(TEST_TEXTS)
    assert np.all(uncertainties >= 0)  # Uncertainties should be non-negative
    
    formatted_examples = estimator._format_examples()
    assert isinstance(formatted_examples, str)
    assert all(ex["text"] in formatted_examples for ex in FEW_SHOT_EXAMPLES)
    assert all(ex["label"] in formatted_examples for ex in FEW_SHOT_EXAMPLES)
