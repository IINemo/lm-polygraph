"""Core utilities with optional vLLM support loaded on demand."""

from lm_polygraph._optional import lazy_exports
from .model import WhiteboxModel, BlackboxModel
from .manager import UEManager

# Keep this eager: the public function shares its name with its submodule.
# Otherwise importing that submodule first replaces the function with a module.
from .estimate_uncertainty import estimate_uncertainty
from .dataset import Dataset
from .api_with_uncertainty import APIWithUncertainty

_EXPORTS = {
    "VLLMWithUncertainty": (".vllm_with_uncertainty", "VLLMWithUncertainty"),
}
__all__ = [
    "WhiteboxModel",
    "BlackboxModel",
    "UEManager",
    "estimate_uncertainty",
    "Dataset",
    "APIWithUncertainty",
    "VLLMWithUncertainty",
]
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
