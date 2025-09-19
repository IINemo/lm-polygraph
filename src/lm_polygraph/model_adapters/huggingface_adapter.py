"""Hugging Face Inference API adapter."""

import json
import logging
import time
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
        # Hugging Face Inference API accepts OpenAI-style payloads for chat completion,
        # so pass parameters through unchanged.
        return params.copy()

    def parse_response(self, response: dict) -> StandardizedResponse:
        """Parse the response into the standardized structure."""
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Invalid Hugging Face response structure: {exc}"
            ) from exc

        return StandardizedResponse(
            text=text,
            logprobs=None,
            finish_reason=response["choices"][0].get("finish_reason"),
            tokens=None,
            raw_response=response,
        )

    def supports_logprobs(self, model_path: str = None) -> bool:
        return False

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

        texts = []

        for prompt in input_texts:
            start = time.time()
            while True:
                current_time = time.time()
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                # tf is going on here, this definitely doesn't work
                response = client.chat_completion(messages)
                output = json.dumps(response, indent=2)

                if isinstance(output, dict):
                    if (list(output.keys())[0] == "error") & (
                        "estimated_time" in output.keys()
                    ):
                        estimated_time = float(output["estimated_time"])
                        elapsed_time = current_time - start
                        print(
                            f"{output['error']}. Estimated time: {round(estimated_time - elapsed_time, 2)} sec."
                        )
                        time.sleep(5)
                    elif (list(output.keys())[0] == "error") & (
                        "estimated_time" not in output.keys()
                    ):
                        log.error(f"{output['error']}")
                        break
                elif isinstance(output, list):
                    break

            texts.append(output[0]["generated_text"])

        return texts
