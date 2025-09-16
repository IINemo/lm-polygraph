"""
Smoke test suite for OpenAI API integration with lm-polygraph

This test mirrors the Together.ai adapter tests to ensure the OpenAI adapter
works correctly with registration, request adaptation, response parsing, and
live API calls when an `OPENAI_API_KEY` is available.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="your-api-key-here"

    # Run the tests
    pytest test/local/test_openai_adapter.py -v

    # Skip smoke tests if no API key is set
    pytest test/local/test_openai_adapter.py -v -k "not openai_api_key"
"""

import os

import pytest

# Import the local OpenAI adapter so it registers itself
import lm_polygraph.model_adapters.openai_adapter  # noqa: F401

from lm_polygraph.model_adapters.blackbox_model import BlackboxModel
from lm_polygraph.estimators import Perplexity
from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
from lm_polygraph.model_adapters.api_provider_adapter import (
    get_adapter,
    list_available_adapters,
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")


@pytest.fixture(scope="module")
def openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


def test_openai_adapter_registration():
    """Ensure the OpenAI adapter is registered and retrievable."""
    available_adapters = list_available_adapters()
    assert (
        "openai" in available_adapters
    ), f"openai adapter not registered. Available: {available_adapters}"

    adapter = get_adapter("openai")
    assert adapter is not None
    assert adapter.__class__.__name__ == "OpenAIAdapter"


@pytest.mark.usefixtures("openai_api_key")
def test_openai_api_smoke(openai_api_key):
    """
    Smoke test for the OpenAI adapter using estimate_uncertainty with Perplexity.

    Validates request formatting, successful API execution, and estimator output.
    """
    test_input = "What is the capital of France?"
    model_name = "gpt-4o-mini"

    model = BlackboxModel(
        model_path=model_name,
        openai_api_key=openai_api_key,
        api_provider_name="openai",
        supports_logprobs=True,
    )

    model.generation_parameters.max_new_tokens = 20
    model.generation_parameters.temperature = 0.0

    estimator = Perplexity()

    result = estimate_uncertainty(
        model=model,
        estimator=estimator,
        input_text=test_input,
    )

    assert result is not None, "estimate_uncertainty returned None"
    assert hasattr(result, "uncertainty"), "Result missing uncertainty field"
    assert hasattr(result, "input_text"), "Result missing input_text field"
    assert hasattr(result, "generation_text"), "Result missing generation_text field"
    assert hasattr(result, "model_path"), "Result missing model_path field"
    assert hasattr(result, "estimator"), "Result missing estimator field"

    assert result.input_text == test_input, f"Input text mismatch: {result.input_text}"
    assert result.model_path == model_name, f"Model path mismatch: {result.model_path}"
    assert result.estimator == str(estimator), f"Estimator mismatch: {result.estimator}"

    assert result.generation_text is not None, "Generation text is None"
    assert len(result.generation_text.strip()) > 0, "Generation text is empty"

    assert isinstance(result.uncertainty, (int, float)), "Uncertainty is not numeric"
    assert not (
        result.uncertainty != result.uncertainty
    ), "Uncertainty is NaN"  # NaN check
    assert result.uncertainty >= 0, "Perplexity should be non-negative"

    print("\n--- OpenAI Smoke Test Results ---")
    print(f"Input: {result.input_text}")
    print(f"Generated: {result.generation_text}")
    print(f"Uncertainty: {result.uncertainty}")
    print(f"Model: {result.model_path}")
    print(f"Estimator: {result.estimator}")
    print(f"Generation tokens: {result.generation_tokens}")


@pytest.mark.usefixtures("openai_api_key")
def test_openai_api_smoke_with_logprobs(openai_api_key):
    """Smoke test verifying logprobs flow using the Perplexity estimator."""
    test_input = "The sky is"
    model_name = "gpt-4o-mini"

    model = BlackboxModel(
        model_path=model_name,
        openai_api_key=openai_api_key,
        api_provider_name="openai",
        supports_logprobs=True,
    )

    model.generation_parameters.max_new_tokens = 5
    model.generation_parameters.temperature = 0.0

    estimator = Perplexity()

    result = estimate_uncertainty(
        model=model,
        estimator=estimator,
        input_text=test_input,
    )

    assert result is not None
    assert result.generation_text is not None
    assert len(result.generation_text.strip()) > 0

    assert isinstance(result.uncertainty, (int, float))
    assert result.uncertainty >= 0

    print("\n--- OpenAI Logprobs Test Results ---")
    print(f"Input: {result.input_text}")
    print(f"Generated: {result.generation_text}")
    print(f"Perplexity: {result.uncertainty}")


def test_openai_adapter_functionality():
    """Directly test adapter parameter adaptation and validation."""
    from lm_polygraph.model_adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()

    test_params = {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "logprobs": 5,
        "top_logprobs": 3,
        "stop_strings": ["\n\n"],
    }

    adapted = adapter.adapt_request(test_params)

    assert "max_tokens" in adapted, "max_new_tokens should be mapped to max_tokens"
    assert adapted["max_tokens"] == 50
    assert "do_sample" not in adapted, "do_sample should be removed"
    assert "stop_strings" not in adapted, "stop_strings should be removed"
    assert adapted["logprobs"] == 5
    assert adapted["top_logprobs"] == 3

    params_with_invalid_ranges = {
        "temperature": 3.5,
        "top_p": 1.5,
        "presence_penalty": 3.0,
        "frequency_penalty": -3.0,
    }

    validated = adapter.validate_parameter_ranges(params_with_invalid_ranges)
    assert validated["temperature"] == 2.0
    assert validated["top_p"] == 1.0
    assert validated["presence_penalty"] == 2.0
    assert validated["frequency_penalty"] == -2.0

    assert adapter.supports_logprobs(), "OpenAI should report logprobs support"


def test_openai_adapter_parse_response_success():
    """Ensure parse_response returns StandardizedResponse with tokens and logprobs."""
    from lm_polygraph.model_adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()

    response = {
        "choices": [
            {
                "message": {"content": "Hello world"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Hello",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "Hello", "logprob": -0.1},
                                {"token": "Hi", "logprob": -1.5},
                            ],
                        },
                        {
                            "token": " world",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": " world", "logprob": -0.2},
                                {"token": " planet", "logprob": -2.0},
                            ],
                        },
                    ]
                },
                "finish_reason": "stop",
            }
        ]
    }

    standardized = adapter.parse_response(response)

    assert standardized.text == "Hello world"
    assert standardized.finish_reason == "stop"
    assert standardized.tokens == ["Hello", " world"]
    assert standardized.logprobs is not None
    assert hasattr(standardized.logprobs, "content")
    assert len(standardized.logprobs.content) == 2
    assert standardized.logprobs.content[0].token == "Hello"


def test_openai_adapter_parse_response_error():
    """Invalid response structures should raise ValueError."""
    from lm_polygraph.model_adapters.openai_adapter import OpenAIAdapter

    adapter = OpenAIAdapter()

    with pytest.raises(ValueError):
        adapter.parse_response({"unexpected": "structure"})
