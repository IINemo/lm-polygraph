"""OpenAI-compatible API Provider Adapter implementations."""

import logging
from typing import Any, List

import openai

from .api_provider_adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)

log = logging.getLogger("lm_polygraph")

class OpenAIChatCompletionMixin:
    """Reusable chat completion inference flow for OpenAI-compatible providers."""

    def _create_client(self, model):
        raise NotImplementedError


    def generate_texts(
        self,
        model,
        input_texts: List[Any],
        args: dict,
    ) -> List[Any]:
        openai_api = self._create_client(model)

        return_logprobs = args.pop("output_scores", False)
        logprobs_args = {}

        if return_logprobs and model.supports_logprobs:
            logprobs_args["logprobs"] = True
            logprobs_args["top_logprobs"] = args.pop("top_logprobs", 5)

        parsed_responses = []
        for prompt in input_texts:
            messages = model.prepare_input(prompt)

            retries = 0
            while True:
                try:
                    response = openai_api.chat.completions.create(
                        model=model.model_path,
                        messages=messages,
                        **args,
                        **logprobs_args,
                    )
                    break
                except Exception as e:
                    if retries > 4:
                        raise Exception from e
                    retries += 1
                    continue

            parsed_responses.append([self.parse_response(resp) for resp in response.choices])

        return parsed_responses


@register_adapter("openai")
class OpenAIAdapter(OpenAIChatCompletionMixin, APIProviderAdapter):
    """
    Adapter for OpenAI API.

    This adapter replicates the exact behavior of the original BlackboxModel
    _validate_args method and response parsing to ensure backward compatibility.
    """

    def adapt_request(self, params: dict) -> dict:
        """
        Adapts parameters for OpenAI API format.
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
            "max_length": "max_completion_tokens",
            "max_new_tokens": "max_completion_tokens",
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
            text = response.message.content

            # Extract logprobs if available
            logprobs = None
            tokens = None
            top_logprobs = None
            alternative_tokens = None
            if hasattr(response, "logprobs") and response.logprobs:
                logprobs_data = response.logprobs
                if hasattr(logprobs_data, "content") and logprobs_data.content:
                    # Extract tokens from logprobs content
                    for item in logprobs_data.content:
                        if hasattr(item, "token"):
                            tokens = tokens or []
                            tokens.append(item.token)

                        if hasattr(item, "logprob"):
                            logprobs = logprobs or []
                            logprobs.append(item.logprob)

                        if hasattr(item, "top_logprobs"):
                            top_logprobs = top_logprobs or []
                            top_logprobs.append([item.logprob for item in item.top_logprobs])

                            alternative_tokens = alternative_tokens or []
                            alternative_tokens.append(
                                [pair.token for pair in item.top_logprobs]
                            )

            # Extract finish reason
            finish_reason = response.finish_reason

            return StandardizedResponse(
                text=text,
                tokens=tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                alternative_tokens=alternative_tokens,
                finish_reason=finish_reason,
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

    def _create_client(self, model):
        return openai.OpenAI()
