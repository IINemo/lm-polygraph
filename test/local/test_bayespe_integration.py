import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.estimators import BayesPEZeroShot, BayesPEFewShot
from lm_polygraph.utils import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel


@pytest.fixture(scope="module")
def model():
    model_id = "sshleifer/tiny-gpt2"
    base_model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return WhiteboxModel(base_model, tokenizer, model_path=model_id)


def test_bayespe_zero_shot_end_to_end(model):
    instructions = [
        "Classify the sentiment of the text.",
        "Is this positive or negative?",
    ]
    class_labels = ["positive", "negative"]
    estimator = BayesPEZeroShot(instructions=instructions, class_labels=class_labels)

    ue = estimate_uncertainty(model, estimator, "I really enjoyed this movie!")

    assert isinstance(ue.uncertainty, float)
    assert ue.input_text == "I really enjoyed this movie!"
    assert ue.estimator.startswith("BayesPEZeroShot")


def test_bayespe_few_shot_end_to_end(model):
    instructions = [
        "Classify the sentiment of the text.",
        "Is this positive or negative?",
    ]
    class_labels = ["positive", "negative"]
    few_shot_examples = [
        {"text": "I love this!", "label": "positive"},
        {"text": "This is terrible.", "label": "negative"},
    ]
    estimator = BayesPEFewShot(
        instructions=instructions,
        few_shot_examples=few_shot_examples,
        class_labels=class_labels,
    )

    ue = estimate_uncertainty(model, estimator, "I really enjoyed this movie!")

    assert isinstance(ue.uncertainty, float)
    assert ue.input_text == "I really enjoyed this movie!"
    assert ue.estimator.startswith("BayesPEFewShot")
