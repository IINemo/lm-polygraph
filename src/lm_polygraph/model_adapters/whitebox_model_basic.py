from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters

from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer



class WhiteboxModelBasic(Model):
    """Basic whitebox model adapter for using in stat calculators and uncertainty estimators."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 tokenizer_args: Dict, parameters: GenerationParameters = GenerationParameters(), model_type=""):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.generation_parameters = parameters
        self.model_type = model_type

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        return self.tokenizer(*args, **self.tokenizer_args, **kwargs)
    
    def device(self):
        """
        Returns the device the model is currently loaded on.

        Returns:
            str: device string.
        """
        return self.model.device

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate_texts(self, input_texts: List[str], **args):
        encoded = self.tokenize(input_texts)
        out = self.generate(encoded, args.pop("args_generate"))
        return self.tokenizer.batch_decode(out["greedy_tokens"])
