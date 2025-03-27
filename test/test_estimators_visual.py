import torch
import pytest

from transformers import AutoModelForVision2Seq, AutoProcessor

from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import *
from lm_polygraph.model_adapters.whitebox_visual import VisualWhiteboxModel

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
