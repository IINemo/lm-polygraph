"""
vLLM logprobs calculator for lm-polygraph.

Extracts greedy_log_likelihoods and greedy_log_probs from vLLM output
or from token_ids/logprobs directly (for truncated scoring).
"""

from typing import Dict, List

import numpy as np
from lm_polygraph.stat_calculators import StatCalculator


class VLLMLogprobsCalculator(StatCalculator):
    """
    Extracts greedy_log_likelihoods and greedy_log_probs from vLLM output
    or from token_ids/logprobs directly.

    Args:
        output_matrix: If True, output greedy_log_probs as 2D matrix [T, K]
                      for PDGap estimator. If False (default), output as
                      list of 1D arrays for EntropyCalculator.

    Usage:
        # From vLLM output (original way)
        deps = {"vllm_output": output}
        result = calculator(deps)

        # From token_ids/logprobs directly (for truncated scoring)
        deps = {"token_ids": truncated_ids, "logprobs": truncated_logprobs}
        result = calculator(deps)
    """

    def __init__(self, output_matrix: bool = False):
        super().__init__()
        self.output_matrix = output_matrix

    @staticmethod
    def meta_info():
        return (
            ["greedy_log_likelihoods", "greedy_log_probs", "greedy_tokens"],
            ["vllm_output"],  # Optional dependency - can also use token_ids/logprobs
        )

    def __call__(self, dependencies: Dict, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract logprobs from vLLM output or from token_ids/logprobs directly.

        Args:
            dependencies: Dict containing either:
                - 'vllm_output': vLLM CompletionOutput object
                OR
                - 'token_ids': List[int] - token IDs
                - 'logprobs': List[Dict] - logprob dicts from vLLM

        Returns:
            Dict with:
            - greedy_log_likelihoods: [[log_likelihood per token]]
            - greedy_log_probs: format depends on output_matrix:
                - False: [[[log_probs per position]]] for EntropyCalculator
                - True: [2D array of shape [T, K]] for PDGap
            - greedy_tokens: [[token_ids]] - generated token IDs
        """
        # Get token_ids and logprobs from either vllm_output or directly
        if "vllm_output" in dependencies:
            output = dependencies["vllm_output"]
            token_ids = output.token_ids
            logprobs = output.logprobs
        elif "token_ids" in dependencies and "logprobs" in dependencies:
            token_ids = dependencies["token_ids"]
            logprobs = dependencies["logprobs"]
        else:
            raise ValueError(
                "VLLMLogprobsCalculator requires either 'vllm_output' or "
                "both 'token_ids' and 'logprobs' in dependencies"
            )

        if not logprobs or not token_ids:
            return {
                "greedy_log_likelihoods": [[]],
                "greedy_log_probs": [np.array([[]]) if self.output_matrix else []],
                "greedy_tokens": [[]],
            }

        # Extract log-likelihood for each chosen token
        log_likelihoods = []
        for token_id, logprob_dict in zip(token_ids, logprobs):
            if logprob_dict is None:
                log_likelihoods.append(-100.0)
            elif token_id in logprob_dict:
                log_likelihoods.append(logprob_dict[token_id].logprob)
            else:
                log_likelihoods.append(-100.0)

        if self.output_matrix:
            # Output as 2D matrix [T, K] for PDGap
            # K = number of logprobs per position (top_k or vocab_size)
            k = len(logprobs[0]) if logprobs and logprobs[0] else 0
            matrix = np.full((len(logprobs), k), -np.inf)
            for t, logprob_dict in enumerate(logprobs):
                if logprob_dict is not None:
                    for i, info in enumerate(logprob_dict.values()):
                        matrix[t, i] = info.logprob
            greedy_log_probs = [matrix]
        else:
            # Output as list of 1D arrays for EntropyCalculator
            greedy_log_probs = []
            for logprob_dict in logprobs:
                if logprob_dict is not None:
                    position_logprobs = np.array(
                        [info.logprob for info in logprob_dict.values()]
                    )
                else:
                    position_logprobs = np.array([])
                greedy_log_probs.append(position_logprobs)
            greedy_log_probs = [greedy_log_probs]

        return {
            "greedy_log_likelihoods": [log_likelihoods],
            "greedy_log_probs": greedy_log_probs,
            "greedy_tokens": [list(token_ids)],
        }
