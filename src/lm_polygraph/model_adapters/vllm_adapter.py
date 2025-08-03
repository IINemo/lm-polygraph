"""
vLLM API Provider Adapter

This adapter handles the vLLM OpenAI-compatible API format, including
vLLM-specific parameters and response parsing.
"""

import logging
from typing import Dict, Any

from lm_polygraph.utils.adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)

log = logging.getLogger("lm_polygraph")


@register_adapter("vllm")
class VLLMAdapter(APIProviderAdapter):
    """
    Adapter for vLLM OpenAI-compatible API.

    Handles vLLM-specific parameters while maintaining OpenAI API compatibility.
    """

    def adapt_request(self, params: dict) -> dict:
        """
        Adapts parameters for vLLM API format.

        vLLM supports additional parameters beyond standard OpenAI API.
        """
        adapted_params = params.copy()

        # vLLM-specific parameter mapping
        if "frequency_penalty" in adapted_params:
            # vLLM uses repetition_penalty instead of frequency_penalty
            adapted_params["repetition_penalty"] = adapted_params.pop(
                "frequency_penalty"
            )

        # Remove parameters that are handled differently or not supported
        for delete_key in [
            "do_sample",
            "min_length",
            "min_new_tokens",
            "num_beams",
            "generate_until",
            "allow_newlines",
        ]:
            adapted_params.pop(delete_key, None)

        # Map HF argument names to vLLM API argument names
        key_mapping = {
            "num_return_sequences": "n",
            "max_length": "max_tokens",
            "max_new_tokens": "max_tokens",
        }
        for key, replace_key in key_mapping.items():
            if key in adapted_params:
                adapted_params[replace_key] = adapted_params[key]
                adapted_params.pop(key)

        return adapted_params

    def parse_response(self, response: dict) -> StandardizedResponse:
        """
        Parses vLLM API response into standardized format.

        vLLM follows OpenAI format, so parsing is similar to OpenAI adapter.

        Args:
            response: Raw vLLM API response dictionary

        Returns:
            StandardizedResponse object
        """
        try:
            # Extract main text content (same as OpenAI format)
            text = response["choices"][0]["message"]["content"]

            # Extract logprobs if available (vLLM supports logprobs)
            logprobs = None
            tokens = None
            if (
                "logprobs" in response["choices"][0]
                and response["choices"][0]["logprobs"]
            ):
                logprobs_data = response["choices"][0]["logprobs"]
                if "content" in logprobs_data and logprobs_data["content"]:
                    # Extract tokens from logprobs content
                    tokens = [
                        item.get("token", "") for item in logprobs_data["content"]
                    ]

                    # Create mock objects for stat calculator compatibility
                    # The stat calculator expects each content item to have .token, .logprob, .top_logprobs attributes
                    class MockTopLogprob:
                        def __init__(self, token, logprob):
                            self.token = token
                            self.logprob = logprob

                    class MockLogprobContent:
                        def __init__(self, item_dict):
                            self.token = item_dict["token"]
                            self.logprob = item_dict["logprob"]
                            self.top_logprobs = []
                            if "top_logprobs" in item_dict:
                                for top_item in item_dict["top_logprobs"]:
                                    self.top_logprobs.append(
                                        MockTopLogprob(
                                            top_item["token"], top_item["logprob"]
                                        )
                                    )

                    class MockLogprobs:
                        def __init__(self, content_list):
                            self.content = [
                                MockLogprobContent(item) for item in content_list
                            ]

                    logprobs = MockLogprobs(logprobs_data["content"])

            # Extract finish reason
            finish_reason = response["choices"][0].get("finish_reason")

            return StandardizedResponse(
                text=text,
                logprobs=logprobs,
                finish_reason=finish_reason,
                tokens=tokens,
                raw_response=response,
            )

        except (KeyError, IndexError, TypeError) as e:
            log.error(f"Error parsing vLLM response: {e}")
            log.error(f"Response structure: {response}")
            raise ValueError(f"Invalid vLLM API response format: {e}")

    def supports_logprobs(self, model_path: str = None) -> bool:
        """
        Check if the vLLM model supports logprobs.

        Args:
            model_path: vLLM model identifier

        Returns:
            True (vLLM generally supports logprobs)
        """
        return True

    def validate_parameter_ranges(self, params: dict) -> dict:
        """
        Validate and clamp parameters to vLLM-specific ranges.

        Args:
            params: Parameters dictionary

        Returns:
            Validated parameters dictionary
        """
        validated_params = params.copy()

        # vLLM parameter ranges (similar to OpenAI with additional parameters)
        parameter_ranges = {
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "top_k": (1, 200),
            "repetition_penalty": (0.0, 2.0),
            "min_p": (0.0, 1.0),
            "presence_penalty": (-2.0, 2.0),
        }

        for param, (min_val, max_val) in parameter_ranges.items():
            if param in validated_params:
                value = validated_params[param]
                if isinstance(value, (int, float)):
                    original_value = value
                    validated_params[param] = max(min_val, min(max_val, value))
                    if original_value != validated_params[param]:
                        log.warning(
                            f"Parameter {param}={original_value} clamped to "
                            f"{validated_params[param]} (range: [{min_val}, {max_val}]) for vLLM"
                        )

        return validated_params
