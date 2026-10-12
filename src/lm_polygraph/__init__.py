"""Public imports, loaded on demand so numerical utilities need no model stack."""

from importlib import import_module

_EXPORTS = {
    "WhiteboxModel": ".utils.model",
    "BlackboxModel": ".utils.model",
    "UEManager": ".utils.manager",
    "estimate_uncertainty": ".utils.estimate_uncertainty",
    "Dataset": ".utils.dataset",
    "APIWithUncertainty": ".utils.api_with_uncertainty",
    "CausalLMWithUncertainty": ".utils.causal_lm_with_uncertainty",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
