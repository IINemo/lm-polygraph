"""Hugging Face Inference API adapter."""

from .api_provider_adapter import (
    APIProviderAdapter,
    StandardizedResponse,
    register_adapter,
)


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
