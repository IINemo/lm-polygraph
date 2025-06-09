"""
vLLM-optimized whitebox model adapter for high-performance uncertainty estimation.

This module provides an adapter for using vLLM (Very Large Language Model) inference
engine with LM-Polygraph's uncertainty estimation framework. vLLM offers significant
speed improvements for inference through techniques like continuous batching and
PagedAttention.
"""

from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters
from transformers.generation import GenerateDecoderOnlyOutput

import torch
from typing import List, Dict, Optional


class WhiteboxModelvLLM(Model):
    """
    Whitebox model adapter optimized for vLLM inference engine.
    
    This adapter enables uncertainty estimation with models deployed using vLLM,
    which provides high-throughput inference for large language models. It maintains
    compatibility with LM-Polygraph's stat calculators and uncertainty estimators
    while leveraging vLLM's performance optimizations.
    
    Key features:
    - High-performance inference with vLLM backend
    - Access to token-level log probabilities
    - Compatible with all whitebox uncertainty methods
    - Automatic handling of vLLM's output format
    - Support for custom sampling parameters
    
    vLLM benefits:
    - Continuous batching for improved throughput
    - PagedAttention for efficient memory usage
    - Optimized CUDA kernels
    - Support for tensor parallelism
    
    Attributes:
        model: vLLM model instance
        tokenizer: Model tokenizer (extracted from vLLM model)
        sampling_params: vLLM SamplingParams configuration
        generation_parameters: LM-Polygraph generation settings
        base_device: Device where model is deployed
        model_type: Set to "vLLMCausalLM"
        
    Examples:
        Basic usage with vLLM:
        >>> from vllm import LLM, SamplingParams
        >>> from lm_polygraph.model_adapters import WhiteboxModelvLLM
        >>> 
        >>> # Initialize vLLM model
        >>> llm = LLM(model="meta-llama/Llama-2-7b-hf")
        >>> sampling_params = SamplingParams(
        ...     temperature=0.8,
        ...     max_tokens=100,
        ...     logprobs=5  # Request top-5 log probabilities
        ... )
        >>> 
        >>> # Create adapter
        >>> model = WhiteboxModelvLLM(llm, sampling_params)
        >>> 
        >>> # Use with uncertainty estimation
        >>> from lm_polygraph.estimators import TokenEntropy
        >>> estimator = TokenEntropy()
        >>> result = estimate_uncertainty(model, estimator, "What is AI?")
        
        With custom generation parameters:
        >>> from lm_polygraph.utils.generation_parameters import GenerationParameters
        >>> gen_params = GenerationParameters(
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     repetition_penalty=1.1
        ... )
        >>> model = WhiteboxModelvLLM(
        ...     llm, sampling_params,
        ...     generation_parameters=gen_params
        ... )
        
    See Also:
        WhiteboxModel: Standard HuggingFace model adapter
        WhiteboxModelBasic: Simplified adapter for basic use cases
        
    Note:
        vLLM must be installed separately: pip install vllm
        Requires GPU with sufficient memory for the model
    """

    def __init__(
        self,
        model,
        sampling_params,
        generation_parameters: GenerationParameters = GenerationParameters(),
        device: str = "cuda",
    ):
        """
        Initialize vLLM model adapter for uncertainty estimation.
        
        Parameters:
            model: vLLM LLM instance, already initialized with desired model
            sampling_params: vLLM SamplingParams object with generation settings.
                Important: Set logprobs > 0 to get uncertainty estimates
            generation_parameters: LM-Polygraph generation parameters that will
                override corresponding vLLM settings
            device: Device identifier where model is deployed. Default: "cuda"
                
        Note:
            The sampling_params.logprobs parameter should be set to a value > 0
            to enable uncertainty estimation. Higher values give access to more
            top tokens but may slightly impact performance.
        """
        self.model = model
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling_params = sampling_params
        self.generation_parameters = generation_parameters

        self.sampling_params.stop = list(
            getattr(self.generation_parameters, "generate_until", list())
        )

        for param in [
            "presence_penalty",
            "repetition_penalty",
            "temperature",
            "top_p",
            "top_k",
        ]:
            setattr(
                self.sampling_params,
                param,
                getattr(self.generation_parameters, param, None),
            )

        self.base_device = device
        self.model_type = "vLLMCausalLM"

    def generate(self, *args, **kwargs):
        """
        Generate text using vLLM's optimized generation pipeline.
        
        Handles conversion between HuggingFace-style inputs and vLLM's
        expected format, then post-processes outputs for compatibility
        with LM-Polygraph's uncertainty estimators.
        
        Parameters:
            *args: Positional arguments (typically not used)
            **kwargs: Keyword arguments including:
                - input_ids: Tokenized input sequences (will be decoded)
                - num_return_sequences: Number of sequences per input
                - Other generation parameters supported by vLLM
                
        Returns:
            GenerateDecoderOnlyOutput: Standard generation output with:
                - sequences: Generated token IDs
                - logits: Token log probabilities for uncertainty estimation
                - scores: Same as logits for compatibility
                
        Examples:
            >>> inputs = model.tokenize(["Hello, world!"])
            >>> outputs = model.generate(**inputs, num_return_sequences=1)
            >>> # outputs.logits contains log probabilities for uncertainty
        """
        sampling_params = self.sampling_params
        sampling_params.n = kwargs.get("num_return_sequences", 1)
        texts = self.tokenizer.batch_decode(
            kwargs["input_ids"], skip_special_tokens=True
        )
        sampling_params.stop = []
        output = self.model.generate(*args, texts, sampling_params)
        return self.post_processing(output)

    def device(self):
        """
        Get the device where the model is deployed.
        
        Returns:
            str: Device identifier (e.g., "cuda", "cuda:0")
        """
        return self.base_device

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts for model generation.
        
        Parameters:
            texts: List of input strings to tokenize
            
        Returns:
            Dict containing:
                - input_ids: Tokenized sequences
                - attention_mask: Attention masks
                - Other tokenizer outputs
                
        Examples:
            >>> tokens = model.tokenize(["Hello!", "How are you?"])
            >>> # Ready for use with generate()
        """
        output = self.tokenizer(texts, return_tensors="pt", padding=True)
        return output

    def __call__(self, *args, **kwargs):
        """
        Direct model call (routes to generate for vLLM compatibility).
        
        Parameters:
            *args: Positional arguments for generate
            **kwargs: Keyword arguments for generate
            
        Returns:
            Generation output from vLLM
        """
        return self.generate(*args, **kwargs)

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        High-level text generation from string inputs.
        
        Parameters:
            input_texts: List of input prompts as strings
            **args: Additional generation arguments
            
        Returns:
            List[str]: Generated texts for each input
            
        Examples:
            >>> texts = model.generate_texts(
            ...     ["Once upon a time", "In a galaxy far away"],
            ...     num_return_sequences=2
            ... )
            >>> # Returns flat list of all generated texts
        """
        outputs = self.generate(input_texts, **args)
        texts = [
            outputs.text
            for sampled_outputs in outputs
            for outputs in sampled_outputs.outputs
        ]
        return texts

    def post_processing(self, outputs):
        """
        Convert vLLM outputs to HuggingFace-compatible format.
        
        Transforms vLLM's RequestOutput format into a GenerateDecoderOnlyOutput
        that's compatible with LM-Polygraph's uncertainty estimators. Extracts
        log probabilities and formats them as logits tensors.
        
        Parameters:
            outputs: vLLM generation outputs
            
        Returns:
            GenerateDecoderOnlyOutput: Standardized output format with:
                - sequences: Token IDs as tensors
                - logits: Log probability tensors for each position
                - scores: Copy of logits for compatibility
                
        Note:
            This method handles padding to ensure all sequences have the same
            length and fills missing probabilities with -inf.
        """
        vocab_size = max(
            self.tokenizer.vocab_size, max(self.tokenizer.added_tokens_decoder.keys())
        )
        logits = []
        sequences = []

        max_seq_len = max(
            [
                len(output.token_ids)
                for sampled_outputs in outputs
                for output in sampled_outputs.outputs
            ]
        )
        for sample_output in outputs:

            for output in sample_output.outputs:

                log_prob = torch.zeros((max_seq_len, vocab_size)).fill_(-torch.inf)
                sequence = (
                    torch.zeros(max_seq_len).fill_(self.tokenizer.eos_token_id).long()
                )

                for i, probs in enumerate(output.logprobs):
                    top_tokens = torch.tensor(list(probs.keys()))
                    top_values = torch.tensor([lp.logprob for lp in probs.values()])
                    log_prob[i, top_tokens] = top_values
                    sequence[i] = output.token_ids[i]

                logits.append(log_prob)
                sequences.append(sequence)

        standard_output = GenerateDecoderOnlyOutput(
            sequences=sequences, logits=logits, scores=logits
        )
        return standard_output
