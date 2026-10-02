"""Lazy public exports and actionable errors for optional dependencies."""

from importlib import import_module

_EXTRAS = {
    "openai": "openai",
    "diskcache": "openai",
    "vllm": "vllm",
    "bitsandbytes": "quantization",
    "sentence_transformers": "semantic",
    "spacy": "nlp",
    "nltk": "nlp",
    "boostedprob": "boostedprob",
    "datasets": "evaluation",
    "pandas": "evaluation",
    "rouge_score": "evaluation",
    "sacrebleu": "evaluation",
    "bert_score": "evaluation",
    "evaluate": "evaluation",
    "absl": "evaluation",
    "comet": "comet",
    "flask": "demo",
    "fastchat": "demo",
}


def _missing_dependency(error, extra=None):
    dependency = (error.name or "").split(".")[0]
    extra = extra or _EXTRAS.get(dependency)
    if extra is None:
        raise error
    raise ImportError(
        f"Optional dependency '{dependency}' is required for this feature. "
        f"Install it with: pip install 'lm-polygraph[{extra}]'"
    ) from error


def require_optional(module, extra):
    """Import a dependency at the point of use, preserving unrelated failures."""
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] == module.split(".")[0]:
            _missing_dependency(error, extra)
        raise


def lazy_exports(package, exports, namespace):
    """Implement PEP 562 exports while preserving names and object identity."""

    def get_attribute(name):
        if name not in exports:
            raise AttributeError(f"module '{package}' has no attribute '{name}'")
        module, attribute = exports[name]
        try:
            value = getattr(import_module(module, package), attribute)
        except ModuleNotFoundError as error:
            _missing_dependency(error)
        namespace[name] = value
        return value

    def directory():
        return sorted(set(namespace) | set(exports))

    return get_attribute, directory
