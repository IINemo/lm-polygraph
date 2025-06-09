"""
Basic whitebox model adapter for simplified uncertainty estimation workflows.

This module provides a lightweight adapter for HuggingFace models that integrates
with LM-Polygraph's stat calculators and uncertainty estimators without the full
feature set of the main WhiteboxModel class.
"""

from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters

from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer


class WhiteboxModelBasic(Model):
    """
    Simplified whitebox model adapter for basic uncertainty estimation tasks.
    
    This adapter provides a minimal interface for using HuggingFace models with
    LM-Polygraph's uncertainty estimation framework. It's designed for cases where
    you need direct access to model internals but don't require the full feature
    set of the main WhiteboxModel class.
    
    Key differences from WhiteboxModel:
    - Simpler interface with fewer features
    - Direct pass-through to underlying model methods
    - No custom stopping criteria or logits processing
    - Lighter weight for basic use cases
    
    Attributes:
        model: The underlying HuggingFace model
        tokenizer: HuggingFace tokenizer
        tokenizer_args: Additional arguments for tokenizer calls
        generation_parameters: Default generation settings
        model_type: Description of model architecture
        
    Examples:
        Basic usage:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = WhiteboxModelBasic(
        ...     base_model, tokenizer,
        ...     tokenizer_args={"return_tensors": "pt", "padding": True}
        ... )
        
        With custom generation parameters:
        >>> from lm_polygraph.utils.generation_parameters import GenerationParameters
        >>> params = GenerationParameters(temperature=0.8, max_new_tokens=50)
        >>> model = WhiteboxModelBasic(
        ...     base_model, tokenizer,
        ...     tokenizer_args={},
        ...     parameters=params
        ... )
        
    See Also:
        WhiteboxModel: Full-featured model adapter with all capabilities
        WhiteboxModelvLLM: Adapter optimized for vLLM inference
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tokenizer_args: Dict,
        parameters: GenerationParameters = GenerationParameters(),
        model_type: str = "",
    ):
        """
        Initialize the basic whitebox model adapter.
        
        Parameters:
            model: Pre-loaded HuggingFace model (CausalLM or Seq2Seq)
            tokenizer: Corresponding HuggingFace tokenizer
            tokenizer_args: Default arguments to pass to tokenizer calls,
                e.g., {"return_tensors": "pt", "padding": True}
            parameters: Default generation parameters. Will be used unless
                overridden in generate calls
            model_type: Optional model type description for logging/debugging
            
        Note:
            The model should already be loaded on the desired device and in
            evaluation mode before passing to this adapter.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_args = tokenizer_args
        self.generation_parameters = parameters
        self.model_type = model_type

    def generate(self, *args, **kwargs):
        """
        Generate model outputs using the underlying model's generate method.
        
        This is a direct pass-through to the HuggingFace model's generate
        method, allowing full access to all generation features.
        
        Parameters:
            *args: Positional arguments for model.generate
            **kwargs: Keyword arguments for model.generate, including:
                - input_ids: Tokenized input sequences
                - attention_mask: Attention mask for inputs
                - max_new_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling
                - etc. (see HuggingFace documentation)
                
        Returns:
            ModelOutput: HuggingFace generation output
            
        Examples:
            >>> inputs = model.tokenize(["Hello, world!"])
            >>> outputs = model.generate(**inputs, max_new_tokens=20)
        """
        return self.model.generate(*args, **kwargs)

    def tokenize(self, *args, **kwargs):
        """
        Tokenize input texts using the model's tokenizer.
        
        Combines the provided arguments with the default tokenizer_args
        specified during initialization.
        
        Parameters:
            *args: Positional arguments for tokenizer (typically text inputs)
            **kwargs: Additional tokenizer arguments that override defaults
            
        Returns:
            BatchEncoding: Tokenized inputs ready for model generation
            
        Examples:
            >>> tokens = model.tokenize(["Hello!", "Hi there!"])
            >>> # Uses default tokenizer_args plus any kwargs
        """
        return self.tokenizer(*args, **self.tokenizer_args, **kwargs)

    def device(self):
        """
        Get the device where the model is currently loaded.
        
        Returns:
            torch.device: Device object (e.g., 'cuda:0', 'cpu')
            
        Examples:
            >>> device = model.device()
            >>> print(f"Model is on: {device}")
        """
        return self.model.device

    def __call__(self, *args, **kwargs):
        """
        Forward pass through the model for getting logits.
        
        This is a direct pass-through to the underlying model's forward
        method, useful for getting raw logits without generation.
        
        Parameters:
            *args: Positional arguments for model forward pass
            **kwargs: Keyword arguments including input_ids, attention_mask, etc.
            
        Returns:
            ModelOutput: Raw model outputs including logits
        """
        return self.model(*args, **kwargs)

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        High-level text generation from string inputs.
        
        Handles the full pipeline of tokenization, generation, and decoding
        for convenience. Note that this method expects 'args_generate' to
        be passed in the kwargs for generation parameters.
        
        Parameters:
            input_texts: List of input prompts as strings
            **args: Must include 'args_generate' dict with generation parameters
            
        Returns:
            List[str]: Generated texts corresponding to each input
            
        Examples:
            >>> texts = model.generate_texts(
            ...     ["Once upon a time"],
            ...     args_generate={"max_new_tokens": 50, "temperature": 0.8}
            ... )
            
        Note:
            This method has a non-standard interface expecting args_generate.
            For standard generation, use tokenize() + generate() + decode().
        """
        encoded = self.tokenize(input_texts)
        out = self.generate(encoded, args.pop("args_generate"))
        return self.tokenizer.batch_decode(out["greedy_tokens"])
