"""Together.ai API Provider Adapter."""

import logging
import os
from typing import Any, List

from together import Together
import numpy as np

from .api_provider_adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)

log = logging.getLogger("lm_polygraph")


class TogetherAIChatCompletionMixin:
    """Reusable chat completion inference flow for TogetherAI-compatible providers."""

    def _create_client(self, model):
        raise NotImplementedError

    def generate_texts(
        self,
        model,
        input_texts: List[Any],
        args: dict,
    ) -> List[Any]:
        together_api = self._create_client(model)

        return_logprobs = args.pop("output_scores", False)
        logprobs_args = {}

        if return_logprobs:
            logprobs_args["logprobs"] = args.pop("top_logprobs", 5)

        parsed_responses = []
        for prompt in input_texts:
            messages = model.prepare_input(prompt)

            retries = 0
            while True:
                try:
                    response = together_api.chat.completions.create(
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

            parsed_responses.append(
                [self.parse_response(resp) for resp in response.choices]
            )
            
            return parsed_responses

@register_adapter("together_ai")
class TogetherAIAdapter(TogetherAIChatCompletionMixin, APIProviderAdapter):
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

                if hasattr(logprobs_data, "content"):
                    # for some models, together.ai  returns logprobs nested under 'content'
                    content_logprobs = logprobs_data.content

                    logprobs = []
                    tokens = []
                    top_logprobs = []
                    alternative_tokens = []
                    for item in content_logprobs:
                        if "token" in item:
                            tokens.append(item["token"])
                        if "logprob" in item:
                            logprobs.append(item["logprob"])
                        if "top_logprobs" in item:
                            tl_dict = item["top_logprobs"]
                            alternative_tokens.append([tl_item['token'] for tl_item in tl_dict])
                            top_logprobs.append([tl_item['logprob'] for tl_item in tl_dict])
                else:
                    if hasattr(logprobs_data, "token_ids"):
                        tokens = logprobs_data.token_ids
                    if hasattr(logprobs_data, "token_logprobs"):
                        logprobs = logprobs_data.token_logprobs
                    if hasattr(logprobs_data, "top_logprobs"):
                        top_logprobs = [
                            list(tl_dict.values()) for tl_dict in logprobs_data.top_logprobs
                        ]
                        alternative_tokens = [
                            list(tl_dict.keys()) for tl_dict in logprobs_data.top_logprobs
                        ]
            
            max_num_lp = np.max([len(lp) for lp in top_logprobs])
            min_num_lp = np.min([len(lp) for lp in top_logprobs])

            if max_num_lp != min_num_lp:
                # clip all to min length for consistency
                top_logprobs = [
                    lp[:min_num_lp] for lp in top_logprobs
                ]
                alternative_tokens = [
                    at[:min_num_lp] for at in alternative_tokens
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
        client = Together(
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )

        return client
