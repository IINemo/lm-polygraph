from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.estimators import BayesPEZeroShot
from lm_polygraph.utils import estimate_uncertainty
from lm_polygraph.utils.model import WhiteboxModel


def test_bayespe_zero_shot_end_to_end():
    model_id = "sshleifer/tiny-gpt2"
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = WhiteboxModel(base_model, tokenizer, model_path=model_id)

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

