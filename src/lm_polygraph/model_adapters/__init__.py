"""Public API, imported on demand to keep optional dependencies optional."""

from lm_polygraph._optional import lazy_exports

_EXPORTS = {
    "WhiteboxModelBasic": (".whitebox_model_basic", "WhiteboxModelBasic"),
    "VisualWhiteboxModel": (".visual_whitebox_model", "VisualWhiteboxModel"),
    "WhiteboxModelvLLM": (".whitebox_model_vllm", "WhiteboxModelvLLM"),
    "WhiteboxModel": ("lm_polygraph.utils.model", "WhiteboxModel"),
    "BlackboxModel": ("lm_polygraph.utils.model", "BlackboxModel"),
}

__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
