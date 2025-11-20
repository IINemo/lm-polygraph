"""Together.ai API Provider Adapter."""

import logging
import os

import openai

from .api_provider_adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)
from .openai_adapter import OpenAIChatCompletionMixin

log = logging.getLogger("lm_polygraph")


@register_adapter("together_ai")
class TogetherAIAdapter(OpenAIChatCompletionMixin, APIProviderAdapter):
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
            "stop_strings",
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
            text = response.message.content

            # Extract logprobs if available - transform to OpenAI format for stat calculator compatibility
            logprobs = None
            tokens = None
            top_logprobs = None
            alternative_tokens = None
            if hasattr(response, "logprobs") and response.logprobs:
                logprobs_data = response.logprobs

                if hasattr(logprobs_data, "tokens"):
                    tokens = logprobs_data.tokens
                if hasattr(logprobs_data, "token_logprobs"):
                    logprobs = logprobs_data.token_logprobs
                if hasattr(logprobs_data, "top_logprobs"):
                    top_logprobs = [
                        list(tl_dict.values()) for tl_dict in logprobs_data.top_logprobs
                    ]
                    alternative_tokens = [
                        list(tl_dict.keys()) for tl_dict in logprobs_data.top_logprobs
                    ]

            return StandardizedResponse(
                text=text,
                tokens=tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                alternative_tokens=alternative_tokens,
                finish_reason=response.finish_reason,
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

    def _create_client(self, model):
        client = openai.OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )

        return client
