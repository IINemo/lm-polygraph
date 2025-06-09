"""
LM-Polygraph: Uncertainty Estimation for Language Models

LM-Polygraph is a comprehensive framework for estimating uncertainty in Large Language
Model (LLM) outputs. It provides state-of-the-art methods to detect hallucinations,
assess generation confidence, and quantify model uncertainty.

Key Features:
- Support for both white-box (full model access) and black-box (API-only) models
- 30+ uncertainty estimation methods across multiple categories
- Unified interface for different model types (HuggingFace, OpenAI, etc.)
- Benchmarking tools for comparing methods
- Both sequence-level and token-level uncertainty estimation

Quick Start:
    >>> from lm_polygraph import WhiteboxModel
    >>> from lm_polygraph.estimators import TokenEntropy
    >>> from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
    >>> 
    >>> # Load a model
    >>> model = WhiteboxModel.from_pretrained("gpt2")
    >>> 
    >>> # Choose an uncertainty estimator
    >>> estimator = TokenEntropy()
    >>> 
    >>> # Estimate uncertainty
    >>> result = estimate_uncertainty(
    ...     model, estimator, 
    ...     "What is the capital of France?"
    ... )
    >>> print(f"Generated: {result.generation_text}")
    >>> print(f"Uncertainty: {result.uncertainty}")

Main Components:
- WhiteboxModel: Wrapper for HuggingFace models with full access
- BlackboxModel: Wrapper for API-based models (OpenAI, etc.)
- estimators: Collection of uncertainty estimation methods
- estimate_uncertainty: High-level function for uncertainty estimation

For detailed documentation, see: https://lm-polygraph.readthedocs.io/
"""

from .utils.model import WhiteboxModel, BlackboxModel
from .utils.manager import UEManager
from .utils.estimate_uncertainty import estimate_uncertainty
from .utils.dataset import Dataset
