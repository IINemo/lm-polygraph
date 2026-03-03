from lm_polygraph.stat_calculators.greedy_probs_blackbox import (
    BlackboxGreedyTextsCalculator,
)
from lm_polygraph.stat_calculators.sample_blackbox import (
    BlackboxSamplingGenerationCalculator,
)


class _DummyOutput:
    def __init__(self, text: str):
        self.text = text


class _DummyBlackboxModel:
    supports_logprobs = False

    def __init__(self):
        self.last_input_texts = None

    def generate_texts(self, input_texts, max_new_tokens=100, n=1, output_scores=False):
        self.last_input_texts = input_texts
        if n == 1:
            return [_DummyOutput("greedy") for _ in input_texts]
        return [[_DummyOutput(f"sample_{i}") for i in range(n)] for _ in input_texts]


def test_blackbox_greedy_calculator_uses_generation_inputs_when_provided():
    model = _DummyBlackboxModel()
    calc = BlackboxGreedyTextsCalculator()
    generation_inputs = [[{"role": "user", "content": "multimodal"}]]

    result = calc(
        {"generation_inputs": generation_inputs},
        texts=["plain-text-prompt"],
        model=model,
        max_new_tokens=5,
    )

    assert model.last_input_texts == generation_inputs
    assert result["greedy_texts"] == ["greedy"]


def test_blackbox_sampling_calculator_uses_generation_inputs_when_provided():
    model = _DummyBlackboxModel()
    calc = BlackboxSamplingGenerationCalculator(samples_n=3)
    generation_inputs = [[{"role": "user", "content": "multimodal"}]]

    result = calc(
        {"generation_inputs": generation_inputs},
        texts=["plain-text-prompt"],
        model=model,
        max_new_tokens=5,
    )

    assert model.last_input_texts == generation_inputs
    assert result["sample_texts"] == [["sample_0", "sample_1", "sample_2"]]

