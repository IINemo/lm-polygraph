"""Exercise the real generation pipeline without downloading model weights."""

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lm_polygraph.estimators import (
    MaximumSequenceProbability,
    MeanTokenEntropy,
    Perplexity,
    TokenEntropy,
)
from lm_polygraph.stat_calculators import InferCausalLMCalculator, EntropyCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import (
    CausalLMWithUncertainty,
    GenerateDecoderOnlyOutputWithUncertainty,
)


@pytest.fixture
def uniform_model():
    vocab = {f"token{i}": i for i in range(12)}
    backend = Tokenizer(WordLevel(vocab, unk_token="token9"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="token9",
        pad_token="token9",
        bos_token="token10",
        eos_token="token11",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    # Zero logits give uniform probabilities and greedy token 0 at every step.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    return model, tokenizer


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("eos_only", [False, True], ids=["length-limit", "eos-only"])
@pytest.mark.parametrize(
    "estimator",
    [MeanTokenEntropy(), Perplexity(), MaximumSequenceProbability(), TokenEntropy()],
    ids=str,
)
def test_causal_lm_with_uncertainty(uniform_model, batch_size, eos_only, estimator):
    model, tokenizer = uniform_model
    if eos_only:
        # Greedy token 0 is now EOS, so there are no generated content tokens.
        model.generation_config.eos_token_id = 0
    llm_with_uncertainty = CausalLMWithUncertainty(
        model,
        tokenizer,
        [InferCausalLMCalculator(tokenize=False), EntropyCalculator()],
        estimator,
    )
    inputs = tokenizer(
        ["token1 token2", "token3 token4"][:batch_size], return_tensors="pt"
    )

    with torch.no_grad():
        output = llm_with_uncertainty.generate(
            input_ids=inputs.input_ids, max_new_tokens=3, do_sample=False
        )

    steps = 1 if eos_only else 3
    assert isinstance(output, GenerateDecoderOnlyOutputWithUncertainty)
    assert output.sequences.shape == (batch_size, 2 + steps)
    torch.testing.assert_close(output.sequences[:, :2], inputs.input_ids)
    torch.testing.assert_close(
        output.sequences[:, 2:], torch.zeros((batch_size, steps), dtype=torch.long)
    )
    assert len(output.scores) == steps
    assert all(score.shape == (batch_size, 12) for score in output.scores)

    # A uniform distribution on 12 tokens has entropy and token NLL log(12).
    if estimator.level == "token":
        assert len(output.uncertainty_score) == batch_size
        for scores in output.uncertainty_score:
            # TokenEntropy currently excludes the final token even at the limit.
            assert scores.shape == (steps - 1,)
            np.testing.assert_allclose(scores, np.log(12.0), rtol=1e-6)
    else:
        assert output.uncertainty_score.shape == (batch_size,)
        expected = np.log(12.0)
        if isinstance(estimator, MaximumSequenceProbability):
            expected *= steps
        np.testing.assert_allclose(output.uncertainty_score, expected, rtol=1e-6)
