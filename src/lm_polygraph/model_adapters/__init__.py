"""
Model adapters for specialized use cases in LM-Polygraph.

This module provides alternative model adapters for specific scenarios beyond
the standard WhiteboxModel and BlackboxModel classes. These adapters enable
uncertainty estimation with specialized inference engines and model types.

Available Adapters:

WhiteboxModelBasic:
    Simplified adapter with minimal overhead for basic uncertainty estimation.
    Best for straightforward use cases that don't need advanced features.
    
WhiteboxModelvLLM:
    High-performance adapter using the vLLM inference engine.
    Provides significant speed improvements for large-scale inference while
    maintaining access to token probabilities for uncertainty estimation.
    
VisualWhiteboxModel:
    Adapter for vision-language models (VLMs) that process both images and text.
    Enables uncertainty estimation for multimodal generation tasks.

Usage Examples:
    Basic adapter:
    >>> from lm_polygraph.model_adapters import WhiteboxModelBasic
    >>> model = WhiteboxModelBasic(base_model, tokenizer, {})
    
    vLLM for high performance:
    >>> from lm_polygraph.model_adapters import WhiteboxModelvLLM
    >>> from vllm import LLM, SamplingParams
    >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
    >>> model = WhiteboxModelvLLM(llm, SamplingParams(logprobs=5))
    
    Visual language models:
    >>> from lm_polygraph.model_adapters import VisualWhiteboxModel
    >>> model = VisualWhiteboxModel.from_pretrained(
    ...     "Salesforce/blip2-opt-2.7b",
    ...     "VisualLM",
    ...     image_paths=["image.jpg"]
    ... )

See individual adapter classes for detailed documentation and examples.
"""

from .whitebox_model_basic import WhiteboxModelBasic
from .whitebox_model_vllm import WhiteboxModelvLLM
from .visual_whitebox_model import VisualWhiteboxModel

__all__ = ["WhiteboxModelBasic", "WhiteboxModelvLLM", "VisualWhiteboxModel"]
