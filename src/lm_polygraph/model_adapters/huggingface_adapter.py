"""Hugging Face Inference API adapter."""

import logging
from typing import Any, List

from huggingface_hub import InferenceClient

from .api_provider_adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)

log = logging.getLogger("lm_polygraph")


@register_adapter("huggingface")
class HuggingFaceAdapter(APIProviderAdapter):
    """Minimal adapter for Hugging Face text generation endpoints."""

    def adapt_request(self, params: dict) -> dict:
        """
        Adapts parameters for HF API format.
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
            "allow_newlines",
        ]:
            args_copy.pop(delete_key, None)

        key_mapping = {
            "num_return_sequences": "n",
            "max_length": "max_tokens",
            "max_new_tokens": "max_tokens",
            "stop_strings": "stop",
        }
        for key, replace_key in key_mapping.items():
            if key in args_copy:
                args_copy[replace_key] = args_copy[key]
                args_copy.pop(key)

        return args_copy

    def parse_response(self, response: dict) -> StandardizedResponse:
        """Parse the response into the standardized structure."""
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
                        top_logprobs.append(
                            [item.logprob for pair in item.top_logprobs]
                        )

                        alternative_tokens = alternative_tokens or []
                        alternative_tokens.append(
                            [pair.token for pair in item.top_logprobs]
                        )

        return StandardizedResponse(
            text=text,
            tokens=tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            alternative_tokens=alternative_tokens,
            finish_reason=response.finish_reason,
            raw_response=response,
        )

    def supports_logprobs(self, model_path: str = None) -> bool:
        return True

    def generate_texts(
        self,
        model,
        input_texts: List[Any],
        args: dict,
    ) -> List[Any]:
        if model.model_path is None:
            raise ValueError(
                "model_path must be specified for Huggingface API inference."
            )

        client = InferenceClient(model=model.model_path)

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
                    response = client.chat_completion(
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
