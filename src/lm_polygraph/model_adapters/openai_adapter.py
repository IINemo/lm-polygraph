"""
OpenAI API Provider Adapter

This adapter handles the OpenAI API format, maintaining backward compatibility
with the existing BlackboxModel implementation.
"""

import logging

from .api_provider_adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)

log = logging.getLogger("lm_polygraph")


@register_adapter("openai")
class OpenAIAdapter(APIProviderAdapter):
    """
    Adapter for OpenAI API.

    This adapter replicates the exact behavior of the original BlackboxModel
    _validate_args method and response parsing to ensure backward compatibility.
    """

    def adapt_request(self, params: dict) -> dict:
        """
        Adapts parameters for OpenAI API format.

        This method contains the exact logic that was previously in
        BlackboxModel._validate_args() to ensure backward compatibility.
        """
        args_copy = params.copy()

        # BlackboxModel specific validation - remove unsupported parameters
        for delete_key in [
            "do_sample",
            "min_length",
            "top_k",
            "repetition_penalty",
            "min_new_tokens",
            "num_beams",
            "stop_strings",
            "allow_newlines",
        ]:
            args_copy.pop(delete_key, None)

        # Map HF argument names to OpenAI API argument names
        key_mapping = {
            "num_return_sequences": "n",
            "max_length": "max_tokens",
            "max_new_tokens": "max_tokens",
        }
        for key, replace_key in key_mapping.items():
            if key in args_copy:
                args_copy[replace_key] = args_copy[key]
                args_copy.pop(key)

        return args_copy

    def parse_response(self, response: dict) -> StandardizedResponse:
        """
        Parses OpenAI API response into standardized format.

        Args:
            response: Raw OpenAI API response dictionary

        Returns:
            StandardizedResponse object
        """
        try:
            # Extract main text content
            text = response["choices"][0]["message"]["content"]

            # Extract logprobs if available
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
            log.error(f"Error parsing OpenAI response: {e}")
            log.error(f"Response structure: {response}")
            raise ValueError(f"Invalid OpenAI API response format: {e}")

    def supports_logprobs(self, model_path: str = None) -> bool:
        """
        Check if the OpenAI model supports logprobs.

        Args:
            model_path: OpenAI model identifier

        Returns:
            True (OpenAI generally supports logprobs)
        """
        return True

    def validate_parameter_ranges(self, params: dict) -> dict:
        """
        Validate and clamp parameters to OpenAI-specific ranges.

        Args:
            params: Parameters dictionary

        Returns:
            Validated parameters dictionary
        """
        validated_params = params.copy()

        # OpenAI parameter ranges
        parameter_ranges = {
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "presence_penalty": (-2.0, 2.0),
            "frequency_penalty": (-2.0, 2.0),
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
                            f"{validated_params[param]} (range: [{min_val}, {max_val}]) for OpenAI"
                        )

        return validated_params
