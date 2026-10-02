from types import SimpleNamespace

import numpy as np
import pytest

from lm_polygraph.stat_calculators.vllm_logprobs_extraction import (
    VLLMLogprobsExtractionCalculator,
)


@pytest.mark.parametrize("output_matrix", [False, True])
@pytest.mark.parametrize("use_vllm_output", [False, True])
def test_vllm_logprobs_extraction(output_matrix, use_vllm_output):
    token_ids = [7, 3]
    logprobs = [
        {2: SimpleNamespace(logprob=-0.1), 7: SimpleNamespace(logprob=-2.3)},
        {3: SimpleNamespace(logprob=-0.4)},
    ]
    if use_vllm_output:
        dependencies = {
            "vllm_output": SimpleNamespace(token_ids=token_ids, logprobs=logprobs)
        }
    else:
        dependencies = {"token_ids": token_ids, "logprobs": logprobs}

    result = VLLMLogprobsExtractionCalculator(output_matrix=output_matrix)(
        dependencies
    )

    assert result["greedy_tokens"] == [[7, 3]]
    np.testing.assert_allclose(result["greedy_log_likelihoods"], [[-2.3, -0.4]])
    assert len(result["greedy_log_probs"]) == 1
    extracted = result["greedy_log_probs"][0]
    if output_matrix:
        np.testing.assert_allclose(extracted, [[-0.1, -2.3], [-0.4, -np.inf]])
    else:
        assert len(extracted) == 2
        np.testing.assert_allclose(extracted[0], [-0.1, -2.3])
        np.testing.assert_allclose(extracted[1], [-0.4])
