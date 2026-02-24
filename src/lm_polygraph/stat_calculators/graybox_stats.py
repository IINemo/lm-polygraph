import numpy as np
from typing import Dict, List, Tuple, Any

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.api_provider_adapter import APIProviderAdapter


class GrayBoxStatsCalculator(StatCalculator):
    """
    Provider-based graybox stats calculator (API models).

    Computes a minimal set of generation statistics for a batch of input texts:

    - greedy_texts:
        The generated text for each input (typically the first returned choice).

    - greedy_tokens_alternatives:
        Per generated token position, the adapter-provided top-k alternatives as
        (token, logprob) pairs.

        Shape:
            List[batch] -> List[position] -> List[(token_str, logprob_float)]
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "greedy_texts",
            "greedy_tokens_alternatives",
        ], []

    def __init__(self, n_alternatives: int = 10):
        super().__init__()
        self.n_alternatives = n_alternatives

    def __call__(
        self,
        dependencies: Dict[str, Any],
        texts: List[str],
        model: APIProviderAdapter,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate continuations for `texts` using an API adapter and store minimal stats.

        Parameters
        ----------
        dependencies: dict used to pass inputs and store computed statistics.
            Must contain:
              - dependencies["model"]: the underlying model wrapper/object used by the adapter
                (e.g., has .prepare_input(...) and .model_path, depending on your design).

        texts: Batch of input prompts.

        model: The APIProviderAdapter instance.

        max_new_tokens:
            Max number of tokens to generate.

        Returns
        -------
        Dict[str, Any]
            The updated dependencies dict with:
              - "greedy_texts"
              - "greedy_tokens_alternatives"
        """
        out = model.generate_texts(
            model=dependencies["model"],
            input_texts=texts,
            args={
                "max_tokens": max_new_tokens,
                "n": 1,
                "output_scores": True,
                "top_logprobs": self.n_alternatives,
            },
        )

        greedy_texts: List[str] = []
        greedy_tokens_alternatives: List[List[List[Tuple[str, float]]]] = []

        # out is expected to be: List[prompt] -> List[choice] -> StandardizedResponse
        for prompt_choices in out:
            if not prompt_choices:
                greedy_texts.append("")
                greedy_tokens_alternatives.append([])
                continue

            # Take the first choice as "greedy" (since n=1)
            resp = prompt_choices[0]

            greedy_texts.append(getattr(resp, "text", "") or "")

            alts_per_pos: List[List[Tuple[str, float]]] = []
            alt_tokens = getattr(resp, "alternative_tokens", None)
            top_lps = getattr(resp, "top_logprobs", None)

            if alt_tokens and top_lps:
                # alt_tokens: List[List[str]]
                # top_lps:    List[List[float]]
                for toks, lps in zip(alt_tokens, top_lps):
                    # Pair token strings with their logprobs
                    alts_per_pos.append(list(zip(toks, lps)))

            greedy_tokens_alternatives.append(alts_per_pos)

        dependencies["greedy_texts"] = greedy_texts
        dependencies["greedy_tokens_alternatives"] = greedy_tokens_alternatives
        return dependencies