from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters

from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer


class WhiteboxModelBasic(Model):
    """Basic whitebox model adapter for using in stat calculators and uncertainty estimators."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tokenizer_args: Dict,
        generation_parameters: GenerationParameters = GenerationParameters(),
        model_type="",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.generation_parameters = generation_parameters
        self.model_type = model_type

    def generate(self, *args, **kwargs):
        """Generates output using the underlying model.

        Args:
            *args: Positional arguments to pass to model.generate()
            **kwargs: Keyword arguments to pass to model.generate(). These will override any
                     matching parameters from self.generation_parameters.

        Returns:
            The output from model.generate() with the combined generation parameters.
        """
        params = self.generation_parameters.copy()
        params.update(kwargs)
        return self.model.generate(*args, **params)

    def tokenize(self, texts: List[str], **kwargs) -> Dict:
        """Tokenizes input texts using the model's tokenizer.
        
        Args:
            texts: List of input text strings to tokenize
            **kwargs: Additional arguments to pass to tokenizer
            
        Returns:
            Dict containing the tokenized inputs
        """
        tokenizer_args = self.tokenizer_args.copy()
        tokenizer_args.update(kwargs)
        return self.tokenizer(texts, **tokenizer_args)

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
