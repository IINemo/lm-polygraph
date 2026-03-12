import pytest

from lm_polygraph.model_adapters.blackbox_model import BlackboxModel


class _DummyResponse:
    def __init__(self, text: str):
        self.text = text


class _DummyAdapter:
    def __init__(self, supports_logprobs: bool):
        self._supports_logprobs = supports_logprobs
        self.supports_logprobs_calls = 0
        self.generate_calls = 0
        self.last_generate_args = None

    def supports_logprobs(self, model_path=None):
        self.supports_logprobs_calls += 1
        return self._supports_logprobs

    def validate_parameter_ranges(self, params: dict) -> dict:
        return params

    def adapt_request(self, params: dict) -> dict:
        return params

    def generate_texts(self, model, input_texts, args):
        self.generate_calls += 1
        self.last_generate_args = args
        return [[_DummyResponse("ok")] for _ in input_texts]


def test_blackbox_model_uses_adapter_support_by_default(monkeypatch):
    adapter = _DummyAdapter(supports_logprobs=False)
    monkeypatch.setattr(
        "lm_polygraph.model_adapters.blackbox_model.get_adapter", lambda _: adapter
    )

    model = BlackboxModel(model_path="dummy-model", api_provider_name="openai")

    assert model.supports_logprobs is False
    assert adapter.supports_logprobs_calls == 1


def test_blackbox_model_override_supports_logprobs_true(monkeypatch):
    adapter = _DummyAdapter(supports_logprobs=False)
    monkeypatch.setattr(
        "lm_polygraph.model_adapters.blackbox_model.get_adapter", lambda _: adapter
    )

    model = BlackboxModel(
        model_path="dummy-model",
        api_provider_name="openai",
        supports_logprobs=True,
    )

    assert model.supports_logprobs is True
    assert adapter.supports_logprobs_calls == 0


def test_blackbox_model_override_supports_logprobs_false_blocks_output_scores(
    monkeypatch,
):
    adapter = _DummyAdapter(supports_logprobs=True)
    monkeypatch.setattr(
        "lm_polygraph.model_adapters.blackbox_model.get_adapter", lambda _: adapter
    )

    model = BlackboxModel(
        model_path="dummy-model",
        api_provider_name="openai",
        supports_logprobs=False,
    )

    with pytest.raises(Exception, match="Cannot access logits"):
        model.generate_texts(input_texts=["prompt"], output_scores=True)

    assert adapter.generate_calls == 0


def test_blackbox_model_override_supports_logprobs_true_allows_output_scores(
    monkeypatch,
):
    adapter = _DummyAdapter(supports_logprobs=False)
    monkeypatch.setattr(
        "lm_polygraph.model_adapters.blackbox_model.get_adapter", lambda _: adapter
    )

    model = BlackboxModel(
        model_path="dummy-model",
        api_provider_name="openai",
        supports_logprobs=True,
    )
    output = model.generate_texts(input_texts=["prompt"], output_scores=True)

    assert adapter.generate_calls == 1
    assert adapter.last_generate_args["output_scores"] is True
    assert output[0][0].text == "ok"
