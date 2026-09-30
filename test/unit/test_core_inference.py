"""Exercise the installed core API with an entirely local, tiny random model."""

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lm_polygraph import WhiteboxModel, estimate_uncertainty
from lm_polygraph.estimators import MeanTokenEntropy
from lm_polygraph.utils.generation_parameters import GenerationParameters


def test_core_inference_without_downloads():
    torch.manual_seed(0)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update({f"word{i}": i + 3 for i in range(16)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=32,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
            attn_implementation="eager",
        )
    ).eval()
    wrapped = WhiteboxModel(
        model,
        tokenizer,
        model_path="tiny-offline",
        generation_parameters=GenerationParameters(max_new_tokens=3),
    )
    result = estimate_uncertainty(wrapped, MeanTokenEntropy(), "word0 word1")
    assert np.isfinite(result.uncertainty)
    assert 0 <= result.uncertainty <= np.log(len(vocab)) + 1e-6
    assert isinstance(result.generation_text, str)
