"""
Smoke test for OpenAI API integration with lm-polygraph

This test verifies that the OpenAI adapter works correctly with actual API calls
using the estimate_uncertainty method and Perplexity estimator.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="your-api-key-here"

    # Run the test
    pytest test/local/test_openai_smoke.py -v

    # Skip if no API key is set
    pytest test/local/test_openai_smoke.py -v -k "not openai_api_key"
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

from lm_polygraph import BlackboxModel, estimate_uncertainty
from lm_polygraph.estimators import Perplexity
from lm_polygraph.model_adapters.api_provider_adapter import (
    get_adapter,
    list_available_adapters,
)


def test_open_ai_adapter_registration():
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


def test_openai_api_smoke_with_logprobs():
    """
    Enhanced smoke test that specifically tests logprobs functionality.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    test_input = "The sky is"
    model_name = "gpt-3.5-turbo"

    # Initialize model with logprobs enabled
    model = BlackboxModel.from_openai(
        openai_api_key=api_key,
        model_path=model_name,
        api_provider_name="openai",
        supports_logprobs=True,
    )

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

    print(f"\n--- OpenAI Logprobs Test Results ---")
    print(f"Input: {result.input_text}")
    print(f"Generated: {result.generation_text}")
    print(f"Perplexity: {result.uncertainty}")
