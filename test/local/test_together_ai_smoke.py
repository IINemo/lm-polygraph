"""
Smoke test for Together.ai API integration with lm-polygraph

This test verifies that the together.ai adapter works correctly with actual API calls
using the estimate_uncertainty method and Perplexity estimator.

Usage:
    # Set your Together.ai API key
    export TOGETHER_API_KEY="your-api-key-here"

    # Run the test
    pytest test/local/test_together_ai_smoke.py -v

    # Skip if no API key is set
    pytest test/local/test_together_ai_smoke.py -v -k "not together_api_key"
"""

import pytest
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")

# Add the source code to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add the test adapters to the Python path
test_adapters_path = Path(__file__).parent / "adapters"
sys.path.insert(0, str(test_adapters_path))

# Import the local together.ai adapter (this registers it)
import together_ai_adapter

from lm_polygraph.utils.model import BlackboxModel
from lm_polygraph.estimators import Perplexity
from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
from lm_polygraph.utils.adapter import get_adapter, list_available_adapters


def test_together_ai_adapter_registration():
    """
    Test that the together.ai adapter is properly registered.
    """
    available_adapters = list_available_adapters()
    assert (
        "together_ai" in available_adapters
    ), f"together_ai adapter not registered. Available: {available_adapters}"

    # Test that we can get the adapter
    adapter = get_adapter("together_ai")
    assert adapter is not None
    assert adapter.__class__.__name__ == "TogetherAIAdapter"


def test_together_ai_api_smoke():
    """
    Smoke test for together.ai API using estimate_uncertainty with Perplexity estimator.

    This test verifies:
    1. Together.ai adapter correctly formats requests
    2. API call succeeds and returns valid response
    3. Perplexity estimator can process the response
    4. estimate_uncertainty returns expected structure
    """
    # Skip test if no API key is provided
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY environment variable not set")

    # Test configuration
    test_input = "What is the capital of France?"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    # Initialize model with together.ai configuration
    model = BlackboxModel(
        model_path=model_name,
        openai_api_key=api_key,
        api_provider_name="together_ai",
        supports_logprobs=True,
    )

    # Override base URL for together.ai (BlackboxModel uses openai client)
    if hasattr(model, "openai_api") and model.openai_api:
        model.openai_api.base_url = "https://api.together.xyz/v1"

    # Use minimal generation parameters for faster test
    model.generation_parameters.max_new_tokens = 20
    model.generation_parameters.temperature = 0.0  # Deterministic for testing

    # Initialize Perplexity estimator
    estimator = Perplexity()

    # Run uncertainty estimation
    result = estimate_uncertainty(
        model=model, estimator=estimator, input_text=test_input
    )

    # Validate result structure
    assert result is not None, "estimate_uncertainty returned None"
    assert hasattr(result, "uncertainty"), "Result missing uncertainty field"
    assert hasattr(result, "input_text"), "Result missing input_text field"
    assert hasattr(result, "generation_text"), "Result missing generation_text field"
    assert hasattr(result, "model_path"), "Result missing model_path field"
    assert hasattr(result, "estimator"), "Result missing estimator field"

    # Validate result values
    assert result.input_text == test_input, f"Input text mismatch: {result.input_text}"
    assert result.model_path == model_name, f"Model path mismatch: {result.model_path}"
    assert result.estimator == str(estimator), f"Estimator mismatch: {result.estimator}"

    # Validate generation text is not empty
    assert result.generation_text is not None, "Generation text is None"
    assert len(result.generation_text.strip()) > 0, "Generation text is empty"

    # Validate uncertainty is a valid number
    assert isinstance(
        result.uncertainty, (int, float)
    ), f"Uncertainty is not numeric: {type(result.uncertainty)}"
    assert not (
        result.uncertainty != result.uncertainty
    ), "Uncertainty is NaN"  # NaN check

    # Basic sanity checks for perplexity
    # Perplexity should be positive (though can be very small)
    assert (
        result.uncertainty >= 0
    ), f"Perplexity should be non-negative, got: {result.uncertainty}"

    # Log results for manual verification
    print(f"\n--- Together.ai Smoke Test Results ---")
    print(f"Input: {result.input_text}")
    print(f"Generated: {result.generation_text}")
    print(f"Uncertainty: {result.uncertainty}")
    print(f"Model: {result.model_path}")
    print(f"Estimator: {result.estimator}")
    print(f"Generation tokens: {result.generation_tokens}")


def test_together_ai_api_smoke_with_logprobs():
    """
    Enhanced smoke test that specifically tests logprobs functionality with 5 alternatives.
    """
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY environment variable not set")

    test_input = "The sky is"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    # Initialize model with logprobs enabled
    model = BlackboxModel(
        model_path=model_name,
        openai_api_key=api_key,
        api_provider_name="together_ai",
        supports_logprobs=True,
    )

    # Override base URL for together.ai
    if hasattr(model, "openai_api") and model.openai_api:
        model.openai_api.base_url = "https://api.together.xyz/v1"

    # Configure for logprobs
    model.generation_parameters.max_new_tokens = 5
    model.generation_parameters.temperature = 0.0

    # Initialize Perplexity estimator (requires logprobs)
    estimator = Perplexity()

    # Run uncertainty estimation
    result = estimate_uncertainty(
        model=model, estimator=estimator, input_text=test_input
    )

    # Validate that we got a valid result
    assert result is not None
    assert result.generation_text is not None
    assert len(result.generation_text.strip()) > 0

    # Perplexity requires logprobs, so this should work
    assert isinstance(result.uncertainty, (int, float))
    assert result.uncertainty >= 0

    print(f"\n--- Together.ai Logprobs Test Results ---")
    print(f"Input: {result.input_text}")
    print(f"Generated: {result.generation_text}")
    print(f"Perplexity: {result.uncertainty}")


def test_together_ai_adapter_functionality():
    """
    Test the together.ai adapter functionality directly.
    """
    # Import and test the adapter directly
    from together_ai_adapter import TogetherAIAdapter

    adapter = TogetherAIAdapter()

    # Test request adaptation
    test_params = {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,  # Should be removed
        "logprobs": 5,
    }

    adapted = adapter.adapt_request(test_params)

    # Check that parameters were adapted correctly
    assert "max_tokens" in adapted, "max_new_tokens should be mapped to max_tokens"
    assert adapted["max_tokens"] == 50
    assert "do_sample" not in adapted, "do_sample should be removed"
    assert adapted["logprobs"] == 5, "logprobs should be preserved"

    # Test parameter validation
    test_params_with_invalid = {
        "temperature": 3.0,  # Should be clamped to 2.0
        "top_k": 500,  # Should be clamped to 200
    }

    validated = adapter.validate_parameter_ranges(test_params_with_invalid)
    assert validated["temperature"] == 2.0, "Temperature should be clamped to 2.0"
    assert validated["top_k"] == 200, "top_k should be clamped to 200"

    # Test logprobs support
    assert adapter.supports_logprobs(), "together.ai should support logprobs"


def test_together_ai_api_error_handling():
    """
    Test error handling for invalid configurations.
    """
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY environment variable not set")

    # Test with invalid model name
    with pytest.raises(Exception):
        model = BlackboxModel(
            model_path="nonexistent-model-12345",
            openai_api_key=api_key,
            api_provider_name="together_ai",
        )

        if hasattr(model, "openai_api") and model.openai_api:
            model.openai_api.base_url = "https://api.together.xyz/v1"

        estimator = Perplexity()
        estimate_uncertainty(model, estimator, "Test input")
