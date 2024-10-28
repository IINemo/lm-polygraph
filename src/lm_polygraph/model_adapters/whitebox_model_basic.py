from lm_polygraph.utils.model import Model

from typing import List


class WhiteboxModelBasic(Model):
    def __init__(self, model, tokenizer):
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
