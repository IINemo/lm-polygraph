from lm_polygraph.utils.model import Model

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer


class WhiteboxModelBasic(Model):
    """Basic whitebox model adapter for using in stat calculators and uncertainty estimators."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate_texts(self, input_texts: List[str], **args):
        encoded = self.tokenize(input_texts)
        out = self.generate(encoded, args.pop("args_generate"))
        return self.tokenizer.batch_decode(out["greedy_tokens"])
