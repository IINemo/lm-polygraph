"""
Together.ai API Provider Adapter for Testing

This adapter handles the together.ai API format, which is highly compatible
with OpenAI's API but uses different base URL and authentication.
"""

import logging
from typing import Dict, Any, Optional

from lm_polygraph.utils.adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)

log = logging.getLogger("lm_polygraph")


@register_adapter("together_ai")
class TogetherAIAdapter(APIProviderAdapter):
    """
    Adapter for Together.ai API.

    Together.ai provides an OpenAI-compatible API with some enhancements
    like additional logprobs features and model-specific parameters.
    """

    def adapt_request(self, params: dict) -> dict:
        """
        Adapts parameters for together.ai API format.

        Together.ai is highly compatible with OpenAI API format,
        so most parameters can be passed through as-is.
        """
        adapted_params = params.copy()

        # Remove parameters that together.ai doesn't support or handles differently
        for delete_key in [
            "do_sample",
            "min_length",
            "min_new_tokens",
            "num_beams",
            "generate_until",
            "allow_newlines",
            "top_k",  # together.ai doesn't accept top_k through OpenAI client
            "repetition_penalty",  # Use frequency_penalty instead
        ]:
            adapted_params.pop(delete_key, None)

        # Map HF argument names to together.ai API argument names
        key_mapping = {
            "num_return_sequences": "n",
            "max_length": "max_tokens",
            "max_new_tokens": "max_tokens",
        }
        for key, replace_key in key_mapping.items():
            if key in adapted_params:
                adapted_params[replace_key] = adapted_params[key]
                adapted_params.pop(key)

        # Handle logprobs parameter conversion for together.ai
        if "logprobs" in adapted_params and adapted_params["logprobs"] is True:
            # Convert OpenAI-style logprobs=True to together.ai logprobs=5
            top_logprobs = adapted_params.pop("top_logprobs", 5)
            adapted_params["logprobs"] = top_logprobs
        elif "top_logprobs" in adapted_params:
            # Remove standalone top_logprobs as together.ai uses logprobs=N
            adapted_params.pop("top_logprobs", None)

        return adapted_params

    def parse_response(self, response: dict) -> StandardizedResponse:
        """
        Parses together.ai API response into standardized format.

        Together.ai follows OpenAI format very closely, with some additional fields.

        Args:
            response: Raw together.ai API response dictionary

        Returns:
            StandardizedResponse object
        """
        try:
            # Extract main text content (same as OpenAI format)
            text = response["choices"][0]["message"]["content"]

            # Extract logprobs if available - transform to OpenAI format for stat calculator compatibility
            logprobs = None
            tokens = None
            if (
                "logprobs" in response["choices"][0]
                and response["choices"][0]["logprobs"]
            ):
                logprobs_data = response["choices"][0]["logprobs"]

                if "tokens" in logprobs_data and "token_logprobs" in logprobs_data:
                    raw_tokens = logprobs_data["tokens"]
                    raw_logprobs = logprobs_data["token_logprobs"]
                    raw_top_logprobs = logprobs_data.get("top_logprobs", [])

                    # Filter out special end-of-text tokens
                    if raw_tokens and raw_tokens[-1] == "<|eot_id|>":
                        raw_tokens = raw_tokens[:-1]
                        raw_logprobs = raw_logprobs[: len(raw_tokens)]
                        if raw_top_logprobs:
                            raw_top_logprobs = raw_top_logprobs[: len(raw_tokens)]

                    tokens = raw_tokens

                    # Create a mock OpenAI-style logprobs object with .content attribute
                    # that the stat calculator can use
                    class MockLogprobContent:
                        def __init__(self, token, logprob, top_logprobs_dict):
                            self.token = token
                            self.logprob = logprob
                            self.top_logprobs = []
                            for tok, prob in top_logprobs_dict.items():
                                mock_top = type(
                                    "MockTopLogprob",
                                    (),
                                    {"token": tok, "logprob": prob},
                                )()
                                self.top_logprobs.append(mock_top)

                    content = []
                    for i, (token, logprob) in enumerate(zip(raw_tokens, raw_logprobs)):
                        top_logprobs_dict = (
                            raw_top_logprobs[i]
                            if i < len(raw_top_logprobs)
                            else {token: logprob}
                        )
                        content.append(
                            MockLogprobContent(token, logprob, top_logprobs_dict)
                        )

                    # Create mock logprobs object with .content attribute
                    class MockLogprobs:
                        def __init__(self, content):
                            self.content = content

                    logprobs = MockLogprobs(content)

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
            log.error(f"Error parsing together.ai response: {e}")
            log.error(f"Response structure: {response}")
            raise ValueError(f"Invalid together.ai API response format: {e}")

    def supports_logprobs(self, model_path: str = None) -> bool:
        """
        Check if the together.ai model supports logprobs.

        Args:
            model_path: Together.ai model identifier

        Returns:
            True (together.ai generally supports logprobs)
        """
        return True

    def validate_parameter_ranges(self, params: dict) -> dict:
        """
        Validate and clamp parameters to together.ai-specific ranges.

        Args:
            params: Parameters dictionary

        Returns:
            Validated parameters dictionary
        """
        validated_params = params.copy()

        # Together.ai parameter ranges (same as OpenAI with some additions)
        parameter_ranges = {
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "top_k": (1, 200),  # Together.ai specific
            "presence_penalty": (-2.0, 2.0),
            "frequency_penalty": (-2.0, 2.0),
            "repetition_penalty": (0.0, 2.0),  # Together.ai specific
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
                            f"{validated_params[param]} (range: [{min_val}, {max_val}]) for together.ai"
                        )

        return validated_params
