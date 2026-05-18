"""
API model wrapper with uncertainty estimation, analogous to VLLMWithUncertainty.

Wraps any OpenAI-compatible API model with lm-polygraph uncertainty scoring.
Supports both generation (delegated to the wrapped model) and standalone scoring
of pre-extracted logprobs.

Usage:
    from lm_polygraph.estimators import MeanTokenEntropy
    from lm_polygraph.stat_calculators import VLLMLogprobsExtractionCalculator, EntropyCalculator
    from lm_polygraph.utils import APIWithUncertainty

    # Wrap an existing API model
    model_with_uncertainty = APIWithUncertainty(
        model=blackbox_model,
        stat_calculators=[VLLMLogprobsExtractionCalculator(), EntropyCalculator()],
        estimator=MeanTokenEntropy(),
    )

    # Option 1: Generate with immediate scoring
    results = model_with_uncertainty.generate(chats, max_new_tokens=1024, n=8)
    # results[i]["uncertainty_score"], results[i]["token_ids"], etc.

    # Option 2: Score pre-extracted logprobs separately
    uncertainty = model_with_uncertainty.score(token_ids, logprobs)

    # Get pseudo-tokenizer for step boundary mapping
    tokenizer = model_with_uncertainty.get_tokenizer()
    tokenizer.set_context(token_ids, logprobs)
    text = tokenizer.decode(token_ids[0:5])
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class APILogprobData:
    """Minimal logprob entry mirroring vLLM's logprob format."""

    logprob: float
    token: str


def convert_api_logprobs(api_logprobs: List[Dict]) -> tuple:
    """Convert OpenAI API logprobs to lm-polygraph/vLLM format.

    API returns: [{token: str, logprob: float, top_logprobs: [{token, logprob}]}]
    lm-polygraph expects: (List[int], List[Dict[int -> obj_with_logprob_attr]])

    Uses hash-based pseudo token IDs since API doesn't provide real IDs.

    Args:
        api_logprobs: List of logprob entries from OpenAI API.

    Returns:
        Tuple of (pseudo_token_ids, vllm_format_logprobs).
    """
    token_ids = []
    logprobs = []

    for entry in api_logprobs:
        main_id = hash(entry["token"]) & 0xFFFFFFFF
        token_ids.append(main_id)
        logprob_dict = {main_id: APILogprobData(entry["logprob"], entry["token"])}
        for top in entry.get("top_logprobs", []):
            tid = hash(top["token"]) & 0xFFFFFFFF
            logprob_dict[tid] = APILogprobData(top["logprob"], top["token"])
        logprobs.append(logprob_dict)

    return token_ids, logprobs


class _APILogprobTokenizer:
    """Pseudo-tokenizer that reconstructs text from API logprob token strings.

    Mirrors the HuggingFace tokenizer.decode() interface used by step boundary
    detection. Since API models use hash-based pseudo token IDs (not real vocab
    IDs), we look up original token text from the logprob dicts instead of
    decoding through a vocabulary.

    Call set_context() with the full trajectory's token_ids and logprobs before
    using decode(). decode() then maps the requested token_ids back to their
    text via the logprob entries.
    """

    def __init__(self):
        self._token_ids = []
        self._logprobs = []
        self._id_to_positions = {}

    def set_context(self, token_ids: List[int], logprobs: List[Dict]):
        """Set the full trajectory context for positional lookup.

        Args:
            token_ids: Full trajectory pseudo token IDs.
            logprobs: Full trajectory logprob dicts (vLLM format).
        """
        self._token_ids = token_ids
        self._logprobs = logprobs
        self._id_to_positions = {}
        for i, tid in enumerate(token_ids):
            self._id_to_positions.setdefault(tid, []).append(i)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Reconstruct text from pseudo token IDs using logprob token strings.

        Args:
            token_ids: Slice of pseudo token IDs to decode.
            skip_special_tokens: Ignored (API tokens don't have special tokens).

        Returns:
            Concatenated token text.
        """
        if not self._logprobs or not token_ids:
            return ""
        texts = []
        for tid in token_ids:
            positions = self._id_to_positions.get(tid, [])
            found = False
            for pos in positions:
                lp_dict = self._logprobs[pos]
                if tid in lp_dict and hasattr(lp_dict[tid], "token"):
                    texts.append(lp_dict[tid].token)
                    found = True
                    break
            if not found:
                texts.append("")
        return "".join(texts)


class APIWithUncertainty:
    """
    Wraps an OpenAI-compatible API model with uncertainty estimation,
    analogous to VLLMWithUncertainty for vLLM models.

    Delegates generation to the wrapped model and scores outputs using
    lm-polygraph stat calculators and estimators. Also supports standalone
    scoring of pre-extracted logprobs via score().

    Args:
        model: API model instance with generate_texts(chats, **kwargs) method
            that returns results with "logprobs" in OpenAI API format.
            Can be None if only using score() for pre-extracted logprobs.
        stat_calculators: List of lm-polygraph stat calculators
            (e.g., [VLLMLogprobsExtractionCalculator(), EntropyCalculator()]).
        estimator: lm-polygraph Estimator instance
            (e.g., MeanTokenEntropy, Perplexity).
    """

    def __init__(self, model=None, stat_calculators: List = None, estimator=None):
        self.model = model
        self.stat_calculators = stat_calculators or []
        self.estimator = estimator
        self._tokenizer = _APILogprobTokenizer()

    def generate(
        self,
        chats: List[List[Dict[str, str]]],
        compute_uncertainty: bool = True,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Generate completions with optional uncertainty scores.

        Delegates to the wrapped model's generate_texts(), converts logprobs
        to vLLM format, and optionally computes uncertainty scores.

        Args:
            chats: List of chat message lists.
            compute_uncertainty: If True, compute uncertainty for all outputs.
            **kwargs: Generation parameters passed to model.generate_texts()
                (max_new_tokens, temperature, n, stop, etc.)

        Returns:
            List of lists of result dicts. Each result dict contains:
                - text: Generated text
                - logprobs: API-format logprobs
                - token_ids: Pseudo token IDs (vLLM-compatible)
                - vllm_logprobs: Logprobs in vLLM format
                - uncertainty_score: Float (if compute_uncertainty=True)
                - finish_reason: API finish reason
        """
        if self.model is None:
            raise ValueError(
                "No model provided. Pass a model to __init__ or use score() directly."
            )

        # Ensure logprobs are requested
        kwargs.setdefault("output_scores", True)

        # Generate via wrapped model
        raw_results = self.model.generate_texts(chats, **kwargs)

        # Process results: convert logprobs and optionally score
        processed_results = []
        for chat_results in raw_results:
            if not isinstance(chat_results, list):
                chat_results = [chat_results]

            processed_chat = []
            for result in chat_results:
                api_logprobs = result.get("logprobs", [])
                token_ids, vllm_logprobs = convert_api_logprobs(api_logprobs)

                processed = {
                    "text": result.get("text", ""),
                    "logprobs": api_logprobs,
                    "token_ids": token_ids,
                    "vllm_logprobs": vllm_logprobs,
                    "finish_reason": result.get("finish_reason", None),
                }

                if compute_uncertainty and token_ids:
                    processed["uncertainty_score"] = self.score(
                        token_ids, vllm_logprobs
                    )
                else:
                    processed["uncertainty_score"] = 0.0

                processed_chat.append(processed)
            processed_results.append(processed_chat)

        return processed_results

    def score(
        self,
        token_ids: List[int],
        logprobs: List[Dict],
    ) -> float:
        """
        Compute uncertainty score from token IDs and logprobs.

        Can be used standalone on pre-extracted logprobs, or called internally
        by generate(). Mirrors VLLMWithUncertainty.score().

        Args:
            token_ids: Pseudo token IDs (from convert_api_logprobs).
            logprobs: Logprob dicts in vLLM-compatible format.

        Returns:
            Uncertainty score (float). Higher = more uncertain.
        """
        if not token_ids or not logprobs:
            return 0.0

        deps = {
            "token_ids": token_ids,
            "logprobs": logprobs,
        }

        for calc in self.stat_calculators:
            deps.update(calc(deps))

        uncertainty = self.estimator(deps)
        return float(uncertainty[0])

    def get_tokenizer(self):
        """Return pseudo-tokenizer for step boundary mapping.

        The returned tokenizer implements decode(token_ids) by looking up
        token text from logprob entries. Call tokenizer.set_context() with
        the full trajectory's token_ids and logprobs before using decode().
        """
        return self._tokenizer

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        if self.model is not None:
            return getattr(self.model, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}' and no model is set"
        )
