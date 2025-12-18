"""
vLLM wrapper with uncertainty estimation, similar to CausalLMWithUncertainty.

Usage:
    from lm_polygraph.estimators import Perplexity, MeanTokenEntropy
    from lm_polygraph.stat_calculators import VLLMLogprobsCalculator, EntropyCalculator
    from vllm import LLM, SamplingParams

    llm = LLM(model="model_path")

    # For Perplexity
    stat_calculators = [VLLMLogprobsCalculator()]
    estimator = Perplexity()

    # For MeanTokenEntropy
    stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
    estimator = MeanTokenEntropy()

    llm_with_uncertainty = VLLMWithUncertainty(llm, stat_calculators, estimator)

    # Option 1: Generate with immediate scoring (scores ALL tokens)
    outputs = llm_with_uncertainty.generate(prompts, sampling_params)

    # Option 2: Generate then score truncated tokens separately
    outputs = llm_with_uncertainty.generate(prompts, sampling_params, compute_uncertainty=False)
    # ... find truncation boundary ...
    uncertainty = llm_with_uncertainty.score(truncated_token_ids, truncated_logprobs)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from lm_polygraph.estimators import Estimator
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


@dataclass
class RequestOutputWithUncertainty:
    """Extends vLLM RequestOutput to include uncertainty scores."""

    request_output: RequestOutput
    uncertainty_scores: List[float]  # One score per output sequence

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped RequestOutput."""
        return getattr(self.request_output, name)


class VLLMWithUncertainty:
    """
    Wraps vLLM LLM with uncertainty estimation using lm-polygraph estimators.

    Similar to CausalLMWithUncertainty but for vLLM backend.

    Args:
        llm: vLLM LLM instance
        stat_calculators: List of stat calculators (e.g., [VLLMLogprobsCalculator(), EntropyCalculator()])
        estimator: lm-polygraph Estimator (e.g., Perplexity, MeanTokenEntropy)
    """

    def __init__(
        self,
        llm: LLM,
        stat_calculators: List,
        estimator: Estimator,
    ):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.stat_calculators = stat_calculators
        self.estimator = estimator

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        compute_uncertainty: bool = True,
    ) -> List[RequestOutputWithUncertainty]:
        """
        Generate completions with optional uncertainty scores.

        Args:
            prompts: Input prompts (single string or list)
            sampling_params: SamplingParams (must have logprobs > 0)
            compute_uncertainty: If True, compute uncertainty for all tokens.
                                If False, return outputs without uncertainty
                                (use score() method later for truncated tokens).

        Returns:
            List of RequestOutputWithUncertainty with uncertainty_scores
            (scores will be empty list if compute_uncertainty=False)
        """
        # Ensure logprobs enabled
        if sampling_params and (sampling_params.logprobs is None or sampling_params.logprobs == 0):
            sampling_params.logprobs = 20

        # Generate with vLLM
        outputs = self.llm.generate(prompts, sampling_params)

        # Compute uncertainty for each request
        results = []
        for request_output in outputs:
            request_scores = []

            if compute_uncertainty:
                for output in request_output.outputs:
                    uncertainty = self.score(output.token_ids, output.logprobs)
                    request_scores.append(uncertainty)

            results.append(
                RequestOutputWithUncertainty(
                    request_output=request_output,
                    uncertainty_scores=request_scores,
                )
            )

        return results

    def score(
        self,
        token_ids: List[int],
        logprobs: List[Dict],
    ) -> float:
        """
        Compute uncertainty score for given tokens and logprobs.

        This method can be used to score truncated tokens separately from generation.
        Useful when you need to truncate at a step boundary before scoring.

        Args:
            token_ids: List of token IDs (can be truncated)
            logprobs: List of logprob dicts from vLLM (can be truncated)

        Returns:
            Uncertainty score (float). Lower = more certain for most estimators.
        """
        if not token_ids or not logprobs:
            return 0.0

        # Build deps dict with token_ids and logprobs directly
        deps = {
            "token_ids": token_ids,
            "logprobs": logprobs,
        }

        # Run stat calculators
        for calc in self.stat_calculators:
            deps.update(calc(deps))

        # Call estimator
        uncertainty = self.estimator(deps)
        return float(uncertainty[0])

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped LLM."""
        return getattr(self.llm, name)
