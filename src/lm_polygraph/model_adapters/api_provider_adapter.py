"""
API Provider Adapter System

This module implements a flexible adapter pattern for handling different API providers
with varying parameter formats and response structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

log = logging.getLogger("lm_polygraph")

# The registry to hold all available adapters
ADAPTER_REGISTRY = {}


def register_adapter(api_provider_name: str):
    """A decorator to register an adapter class."""

    def decorator(cls):
        ADAPTER_REGISTRY[api_provider_name] = cls
        return cls

    return decorator


@dataclass
class StandardizedResponse:
    """A unified data structure for API responses."""

    text: str
    tokens: Optional[List[str]] = None
    logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[List[float]]] = None
    alternative_tokens: Optional[List[List[str]]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class APIProviderAdapter(ABC):
    """
    Abstract base class for API Provider Adapters.

    Each adapter is responsible for:
    1. Translating request parameters to provider-specific format
    2. Parsing provider-specific responses to standardized format
    """

    @abstractmethod
    def adapt_request(self, params: dict) -> dict:
        """
        Adapts the generation parameters for the specific API provider.

        Args:
            params: Dictionary of generation parameters

        Returns:
            Modified parameters dictionary suitable for the provider
        """
        pass

    @abstractmethod
    def parse_response(self, response: dict) -> StandardizedResponse:
        """
        Parses the raw API response into the StandardizedResponse format.

        Args:
            response: Raw API response dictionary

        Returns:
            StandardizedResponse object
        """
        pass

    @abstractmethod
    def generate_texts(
        self,
        model,
        input_texts: List[Any],
        args: Dict[str, Any],
    ) -> List[Any]:
        """Execute provider-specific inference.

        Args:
            model: The calling BlackboxModel instance for state updates.
            input_texts: List of prompts/messages to generate for.
            args: Provider-specific generation parameters.

        Returns:
            List of generated outputs matching the provider's format.
        """
        pass

    def supports_logprobs(self, model_path: str = None) -> bool:
        """
        Check if the provider/model supports logprobs.

        Args:
            model_path: Optional model identifier for model-specific checks

        Returns:
            True if logprobs are supported, False otherwise
        """
        return False

    def validate_parameter_ranges(self, params: dict) -> dict:
        """
        Validate and clamp parameters to provider-specific ranges.

        Args:
            params: Parameters dictionary

        Returns:
            Validated parameters dictionary
        """
        return params


def get_adapter(api_provider_name: str) -> APIProviderAdapter:
    """
    Factory function to get an adapter instance from the registry.

    Args:
        api_provider_name: Name of the API provider

    Returns:
        APIProviderAdapter instance

    Defaults to 'openai' if the requested provider is not found.
    """
    if api_provider_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"No adapter found for provider '{api_provider_name}'. Available providers: {list(ADAPTER_REGISTRY.keys())}"
        )

    adapter_class = ADAPTER_REGISTRY[api_provider_name]

    return adapter_class()
