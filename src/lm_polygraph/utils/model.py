import requests
import torch
import openai
import time
import logging

from dataclasses import asdict
from typing import List, Dict, Optional, Union
from abc import abstractmethod, ABC
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    BartForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizer,
)

from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.ensemble_utils.ensemble_generator import EnsembleGenerationMixin
from lm_polygraph.utils.ensemble_utils.dropout import replace_dropout

log = logging.getLogger("lm_polygraph")


class Model(ABC):
    """
    Abstract model class. Used as base class for both White-box models and Black-box models.
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Parameters:
            model_path (str): unique model path where it can be found.
            model_type (str): description of additional model properties. Can be 'Blackbox' or model specifications
                in the case of white-box.
        """
        self.model_path = model_path
        self.model_type = model_type

    @abstractmethod
    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Abstract method. Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def generate(self, **args):
        """
        Abstract method. Generates the model output with scores from batch formed by HF Tokenizer.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, **args):
        """
        Abstract method. Calls the model on the input batch. Returns the resulted scores.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")


class BlackboxModel(Model):
    """
    Black-box model wrapper for LLMs without access to internal scores and logits.
    
    This class provides a unified interface for uncertainty estimation with models
    accessed through APIs (OpenAI, HuggingFace Inference API) or other services
    where internal model states are not available. Despite the black-box nature,
    some uncertainty estimation methods can still be applied using multiple
    generations or analyzing output patterns.
    
    The class supports:
    - OpenAI API models (GPT-3.5, GPT-4, etc.)
    - HuggingFace Inference API models
    - Models with optional logprob support (e.g., newer OpenAI models)
    - Custom generation parameters per request
    
    Limitations:
    - Cannot access token probabilities (unless supports_logprobs=True)
    - Cannot use white-box uncertainty methods
    - Limited to methods like LexicalSimilarity, NumSemSets, EigValLaplacian
    
    Attributes:
        model_path (str): Model identifier (OpenAI model name or HF model path)
        model_type (str): Always "Blackbox" for this class
        generation_parameters (GenerationParameters): Default generation settings
        openai_api_key (str): API key for OpenAI models
        hf_api_token (str): API token for HuggingFace models
        supports_logprobs (bool): Whether the model API returns log probabilities
        
    Examples:
        OpenAI model usage:
        >>> from lm_polygraph import BlackboxModel
        >>> model = BlackboxModel.from_openai(
        ...     'YOUR_OPENAI_TOKEN',
        ...     'gpt-3.5-turbo'
        ... )
        >>> texts = model.generate_texts(["What is AI?"])
        
        OpenAI with logprobs support:
        >>> model = BlackboxModel.from_openai(
        ...     'YOUR_OPENAI_TOKEN',
        ...     'gpt-4',
        ...     supports_logprobs=True  # Enable for compatible models
        ... )
        
        HuggingFace model usage:
        >>> model = BlackboxModel.from_huggingface(
        ...     hf_api_token='YOUR_API_TOKEN',
        ...     hf_model_id='google/flan-t5-base'
        ... )
        
        With custom generation parameters:
        >>> from lm_polygraph.utils.generation_parameters import GenerationParameters
        >>> params = GenerationParameters(temperature=0.7, max_new_tokens=100)
        >>> model = BlackboxModel.from_openai(
        ...     'YOUR_TOKEN', 'gpt-3.5-turbo',
        ...     generation_parameters=params
        ... )
    
    See Also:
        WhiteboxModel: For models with full access to logits
        GenerationParameters: For configuring generation behavior
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model_path: str = None,
        hf_api_token: str = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
        supports_logprobs: bool = False,
    ):
        """
        Parameters:
            openai_api_key (Optional[str]): OpenAI API key if the blackbox model comes from OpenAI. Default: None.
            model_path (Optional[str]): Unique model path. Openai model name, if `openai_api_key` is specified,
                huggingface path, if `hf_api_token` is specified. Default: None.
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            generation_parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
            supports_logprobs (bool): Whether the model supports returning log probabilities. Default: False.
        """
        super().__init__(model_path, "Blackbox")
        self.generation_parameters = generation_parameters
        self.openai_api_key = openai_api_key
        self.supports_logprobs = supports_logprobs

        if openai_api_key is not None:
            self.openai_api = openai.OpenAI(api_key=openai_api_key)

        self.hf_api_token = hf_api_token

    def _validate_args(self, args):
        """
        Validates and adapts arguments for BlackboxModel generation.

        Parameters:
            args (dict): The arguments to validate.

        Returns:
            dict: Validated and adapted arguments.
        """
        args_copy = args.copy()

        # BlackboxModel specific validation
        for delete_key in [
            "do_sample",
            "min_length",
            "top_k",
            "repetition_penalty",
            "min_new_tokens",
            "num_beams",
            "generate_until",
            "allow_newlines",
        ]:
            args_copy.pop(delete_key, None)

        # Map HF argument names to OpenAI/HF API argument names
        key_mapping = {
            "num_return_sequences": "n",
            "max_length": "max_tokens",
            "max_new_tokens": "max_tokens",
        }
        for key, replace_key in key_mapping.items():
            if key in args_copy:
                args_copy[replace_key] = args_copy[key]
                args_copy.pop(key)

        return args_copy

    def _query(self, payload):
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_path}"
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    @staticmethod
    def from_huggingface(hf_api_token: str, hf_model_id: str, **kwargs):
        """
        Initializes a blackbox model from huggingface.

        Parameters:
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            hf_model_id (Optional[str]): model path in huggingface.
        """
        generation_parameters = kwargs.pop(
            "generation_parameters", GenerationParameters()
        )
        return BlackboxModel(
            hf_api_token=hf_api_token,
            model_path=hf_model_id,
            generation_parameters=generation_parameters,
        )

    @staticmethod
    def from_openai(
        openai_api_key: str, model_path: str, supports_logprobs: bool = False, **kwargs
    ):
        """
        Initializes a blackbox model from OpenAI API.

        Parameters:
            openai_api_key (Optional[str]): OpenAI API key. Default: None.
            model_path (Optional[str]): model name in OpenAI.
            supports_logprobs (bool): Whether the model supports returning log probabilities. Default: False.
        """
        generation_parameters = kwargs.pop(
            "generation_parameters", GenerationParameters()
        )
        return BlackboxModel(
            openai_api_key=openai_api_key,
            model_path=model_path,
            supports_logprobs=supports_logprobs,
            generation_parameters=generation_parameters,
        )

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generate text completions for a batch of input texts.
        
        High-level text generation method that handles tokenization, generation,
        and decoding automatically. This is the primary method for getting text
        outputs from the model.
        
        Parameters:
            input_texts: List of input prompts as strings
            **args: Generation parameters that override defaults:
                - max_new_tokens (int): Maximum new tokens to generate
                - temperature (float): Sampling temperature (0.0 = greedy)
                - top_k (int): Top-k sampling parameter
                - top_p (float): Nucleus sampling parameter  
                - do_sample (bool): Use sampling instead of greedy decoding
                - num_beams (int): Beam search width (1 = no beam search)
                - repetition_penalty (float): Penalty for repeated tokens
                - num_return_sequences (int): Sequences per input
                - pad_token_id (int): Override padding token ID
                - eos_token_id (int): Override end-of-sequence token ID
                
        Returns:
            List[str]: Generated texts corresponding to each input prompt.
                If num_return_sequences > 1, returns List[List[str]].
                
        Examples:
            Simple generation:
            >>> texts = model.generate_texts(["Once upon a time"])
            >>> print(texts[0])
            
            Sampling with temperature:
            >>> texts = model.generate_texts(
            ...     ["Explain quantum computing"],
            ...     do_sample=True,
            ...     temperature=0.8,
            ...     max_new_tokens=100
            ... )
            
            Multiple sequences per input:
            >>> texts = model.generate_texts(
            ...     ["Write a haiku about AI"],
            ...     do_sample=True,
            ...     num_return_sequences=3
            ... )
            >>> # texts[0] contains 3 different haikus
            
        Note:
            - Automatically handles batch padding
            - Applies model-specific token fixing if needed
            - Uses default generation parameters unless overridden
            - For token-level access, use the generate() method instead
        """
        # Apply default parameters first, then override with provided args
        default_params = asdict(self.generation_parameters)
        default_params.update(args)
        args = self._validate_args(default_params)

        # Check if we're trying to access features that require logprobs support
        if (
            any(
                args.get(arg, False)
                for arg in [
                    "output_scores",
                    "output_attentions",
                    "output_hidden_states",
                ]
            )
            and not self.supports_logprobs
        ):
            raise Exception("Cannot access logits for blackbox model")

        texts = []

        if self.openai_api_key is not None:
            # Save log probabilities if requested
            self.last_response = None
            self.logprobs = []
            self.tokens = []

            # Check if we need to return logprobs
            return_logprobs = args.pop("output_scores", False)
            logprobs_args = {}

            if return_logprobs and self.supports_logprobs:
                logprobs_args["logprobs"] = True
                # OpenAI supports returning top logprobs, default to 5
                logprobs_args["top_logprobs"] = args.pop("top_logprobs", 5)

            for prompt in input_texts:
                if isinstance(prompt, str):
                    # If prompt is a string, create a single message with "user" role
                    messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list) and all(
                    isinstance(item, dict) for item in prompt
                ):
                    # If prompt is a list of dictionaries, assume it's already structured as chat
                    messages = prompt
                else:
                    raise ValueError(
                        "Invalid prompt format. Must be either a string or a list of dictionaries."
                    )

                retries = 0
                while True:
                    try:
                        response = self.openai_api.chat.completions.create(
                            model=self.model_path,
                            messages=messages,
                            **args,
                            **logprobs_args,
                        )
                        break
                    except Exception as e:
                        if retries > 4:
                            raise Exception from e
                        else:
                            retries += 1
                            continue

                if args.get("n", 1) == 1:
                    texts.append(response.choices[0].message.content)
                    # Store logprobs if available
                    if return_logprobs and hasattr(response.choices[0], "logprobs"):
                        self.logprobs.append(response.choices[0].logprobs)
                        # Extract token information if available
                        if hasattr(response.choices[0].logprobs, "content"):
                            tokens = [
                                item.token
                                for item in response.choices[0].logprobs.content
                            ]
                            self.tokens.append(tokens)
                else:
                    texts.append([resp.message.content for resp in response.choices])
                    # For multiple returns, we don't collect logprobs for now

                # Store the last response for later use
                self.last_response = response

        elif (self.hf_api_token is not None) & (self.model_path is not None):
            for prompt in input_texts:
                start = time.time()
                while True:
                    current_time = time.time()
                    output = self._query({"inputs": prompt})

                    if isinstance(output, dict):
                        if (list(output.keys())[0] == "error") & (
                            "estimated_time" in output.keys()
                        ):
                            estimated_time = float(output["estimated_time"])
                            elapsed_time = current_time - start
                            print(
                                f"{output['error']}. Estimated time: {round(estimated_time - elapsed_time, 2)} sec."
                            )
                            time.sleep(5)
                        elif (list(output.keys())[0] == "error") & (
                            "estimated_time" not in output.keys()
                        ):
                            log.error(f"{output['error']}")
                            break
                    elif isinstance(output, list):
                        break

                texts.append(output[0]["generated_text"])
        else:
            print(
                "Please provide HF API token and model id for using models from HF or openai API key for using OpenAI models"
            )

        return texts

    def generate(self, **args):
        """
        For OpenAI models with logprobs support, returns a lightweight wrapper around OpenAI API response.
        For other blackbox models, raises an exception as this is not implemented.

        Parameters:
            **args: Arguments to pass to the generate method.
        Returns:
            object: A wrapper around the OpenAI API response if logprobs are supported.
        Raises:
            Exception: If the model doesn't support logprobs.
        """
        if self.supports_logprobs:
            # Apply default parameters first, then override with provided args
            default_params = asdict(self.generation_parameters)
            default_params.update(args)
            args = self._validate_args(default_params)

            args["output_scores"] = True
            sequences = self.generate_texts(**args)

            # Return a simple object with the necessary attributes for compatibility
            class OpenAIGenerationOutput:
                def __init__(self, sequences, scores):
                    self.sequences = sequences
                    self.scores = scores

            return OpenAIGenerationOutput(sequences, self.logprobs)
        else:
            raise Exception("Cannot access logits of blackbox model")

    def __call__(self, **args):
        """
        Not implemented for blackbox models.
        """
        raise Exception("Cannot access logits of blackbox model")

    def tokenizer(self, *args, **kwargs):
        """
        Not implemented for blackbox models.
        """
        raise Exception("Cannot access logits of blackbox model")


class WhiteboxModel(Model):
    """
    White-box model wrapper for HuggingFace models with full access to logits and hidden states.
    
    This class provides a unified interface for uncertainty estimation methods that require
    access to model internals such as token probabilities, attention weights, and hidden states.
    It wraps HuggingFace transformers models and provides additional functionality for
    uncertainty quantification research and applications.
    
    The class supports:
    - All HuggingFace CausalLM models (GPT, LLaMA, Falcon, etc.)
    - All HuggingFace Seq2Seq models (T5, BART, etc.)
    - Custom generation parameters and stopping criteria
    - Ensemble generation for uncertainty estimation
    - Access to raw logits and attention scores
    - Multiple decoding strategies (greedy, sampling, beam search)
    
    Key features:
    - Full access to model logits and probabilities
    - Support for custom generation parameters
    - Efficient batch processing
    - Compatible with all white-box uncertainty estimation methods
    - Automatic handling of special tokens
    
    Attributes:
        model: The underlying HuggingFace model
        tokenizer: HuggingFace tokenizer for the model
        model_path (str): Path or identifier of the model
        model_type (str): Type of model architecture ('CausalLM' or 'Seq2Seq')
        generation_parameters: Default parameters for text generation
        
    Examples:
        Basic initialization from pretrained:
        >>> from lm_polygraph import WhiteboxModel
        >>> model = WhiteboxModel.from_pretrained("gpt2")
        >>> text = model.generate_texts(["Hello, my name is"])[0]
        
        With custom device and parameters:
        >>> model = WhiteboxModel.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     device_map="auto",
        ...     load_in_8bit=True,
        ...     generation_params={"temperature": 0.7, "max_new_tokens": 100}
        ... )
        
        Using existing HuggingFace objects:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = WhiteboxModel(base_model, tokenizer, model_path="gpt2")
        
        For uncertainty estimation:
        >>> from lm_polygraph.estimators import TokenEntropy
        >>> estimator = TokenEntropy()
        >>> uncertainty = estimate_uncertainty(model, estimator, "Explain quantum physics")
        
    See Also:
        BlackboxModel: For models without logit access
        GenerationParameters: For configuring generation behavior
        create_ensemble: For creating ensemble models
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_path: str = None,
        model_type: str = "CausalLM",
        generation_parameters: GenerationParameters = GenerationParameters(),
    ):
        """
        Parameters:
            model (AutoModelForCausalLM): HuggingFace model.
            tokenizer (AutoTokenizer): HuggingFace tokenizer.
            model_path (Optional[str]): Unique model path in HuggingFace.
            model_type (str): Additional model specifications.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
        """
        super().__init__(model_path, model_type)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_parameters = generation_parameters

    def _validate_args(self, args):
        """
        Validates and adapts arguments for WhiteboxModel generation.

        Parameters:
            args (dict): The arguments to validate.

        Returns:
            dict: Validated and adapted arguments.
        """
        args_copy = args.copy()

        # WhiteboxModel specific validation
        if "presence_penalty" in args_copy and args_copy["presence_penalty"] != 0.0:
            log.warning(
                "Skipping requested argument presence_penalty={}".format(
                    args_copy["presence_penalty"]
                )
            )

        # Remove arguments that are not supported by the HF model.generate function
        keys_to_remove = ["presence_penalty", "generate_until", "allow_newlines"]
        for key in keys_to_remove:
            args_copy.pop(key, None)

        return args_copy

    class _ScoresProcessor:
        # Stores original token scores instead of the ones modified with generation parameters
        def __init__(self):
            self.scores = []

        def __call__(self, input_ids=None, scores=None):
            self.scores.append(scores.log_softmax(-1))
            return scores

    class _MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence.
        Copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/utils.py#L208
        """

        def __init__(
            self,
            sequence: str,
            tokenizer: PreTrainedTokenizer,
            initial_decoder_input_length: int,
            batch_size: int,
        ) -> None:
            self.initial_decoder_input_length = initial_decoder_input_length
            self.done_tracker = [False] * batch_size
            self.sequence = sequence
            self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
            # print(sequence, self.sequence_ids)
            # we look back for 2 more tokens than it takes to encode our stop sequence
            # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
            # and we don't want to mistakenly not stop a generation because our
            # (string) stop sequence was output in a different tokenization

            # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
            # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
            # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
            self.sequence_id_len = len(self.sequence_ids) + 2
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
            lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

            lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

            for i, done in enumerate(self.done_tracker):
                if not done:
                    self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
            return False not in self.done_tracker

    def get_stopping_criteria(self, input_ids: torch.Tensor):
        eos = self.tokenizer.decode(self.tokenizer.eos_token_id)
        stop_sequences = self.generation_parameters.generate_until + [eos]
        return StoppingCriteriaList(
            [
                *[
                    self._MultiTokenEOSCriteria(
                        sequence, self.tokenizer, input_ids.shape[1], input_ids.shape[0]
                    )
                    for sequence in stop_sequences
                ],
            ]
        )

    def generate(self, **args):
        """
        Generate model outputs with full access to scores and internal states.
        
        This method provides low-level generation with access to token probabilities,
        attention weights, and hidden states. It's primarily used by uncertainty
        estimation methods that need detailed model outputs.
        
        Parameters:
            input_ids (torch.Tensor): Tokenized input sequences
            attention_mask (torch.Tensor, optional): Attention mask for inputs
            **args: HuggingFace generation parameters:
                - max_new_tokens (int): Maximum tokens to generate
                - min_new_tokens (int): Minimum tokens to generate
                - temperature (float): Sampling temperature
                - top_k (int): Top-k sampling parameter
                - top_p (float): Nucleus sampling parameter
                - do_sample (bool): Whether to use sampling
                - num_beams (int): Number of beams for beam search
                - num_return_sequences (int): Sequences to generate per input
                - output_scores (bool): Return token probabilities
                - output_attentions (bool): Return attention weights
                - output_hidden_states (bool): Return hidden states
                - return_dict_in_generate (bool): Return ModelOutput object
                - stopping_criteria: Custom stopping criteria
                - logits_processor: Custom logits processors
                
        Returns:
            ModelOutput: HuggingFace generation output containing:
                - sequences: Generated token IDs
                - scores: Token probabilities (if output_scores=True)
                - attentions: Attention weights (if output_attentions=True)
                - hidden_states: Hidden states (if output_hidden_states=True)
                
        Examples:
            Basic generation:
            >>> inputs = model.tokenizer("Hello", return_tensors="pt")
            >>> outputs = model.generate(**inputs, max_new_tokens=20)
            >>> text = model.tokenizer.decode(outputs.sequences[0])
            
            With scores for uncertainty:
            >>> outputs = model.generate(
            ...     **inputs,
            ...     max_new_tokens=50,
            ...     output_scores=True,
            ...     do_sample=True,
            ...     temperature=0.8
            ... )
            >>> # Access token probabilities in outputs.scores
            
        Note:
            - This method preserves original token probabilities before any
              processing by generation parameters
            - Custom stopping criteria from generation_parameters are applied
            - The method is primarily for internal use by estimators
        """
        default_params = asdict(self.generation_parameters)

        if len(self.generation_parameters.generate_until) > 0:
            args["stopping_criteria"] = self.get_stopping_criteria(args["input_ids"])

        # add ScoresProcessor to collect original scores
        processor = self._ScoresProcessor()
        if "logits_processor" in args.keys():
            logits_processor = LogitsProcessorList(
                [processor, args["logits_processor"]]
            )
        else:
            logits_processor = LogitsProcessorList([processor])
        args["logits_processor"] = logits_processor

        # update default parameters with passed arguments
        default_params.update(args)
        args = default_params
        args = self._validate_args(args)

        generation = self.model.generate(**args)

        # override generation.scores with original scores from model
        generation.generation_scores = generation.scores
        generation.scores = processor.scores

        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generate text completions for a batch of input texts.
        
        High-level text generation method that handles tokenization, generation,
        and decoding automatically. This is the primary method for getting text
        outputs from the model.
        
        Parameters:
            input_texts: List of input prompts as strings
            **args: Generation parameters that override defaults:
                - max_new_tokens (int): Maximum new tokens to generate
                - temperature (float): Sampling temperature (0.0 = greedy)
                - top_k (int): Top-k sampling parameter
                - top_p (float): Nucleus sampling parameter  
                - do_sample (bool): Use sampling instead of greedy decoding
                - num_beams (int): Beam search width (1 = no beam search)
                - repetition_penalty (float): Penalty for repeated tokens
                - num_return_sequences (int): Sequences per input
                - pad_token_id (int): Override padding token ID
                - eos_token_id (int): Override end-of-sequence token ID
                
        Returns:
            List[str]: Generated texts corresponding to each input prompt.
                If num_return_sequences > 1, returns List[List[str]].
                
        Examples:
            Simple generation:
            >>> texts = model.generate_texts(["Once upon a time"])
            >>> print(texts[0])
            
            Sampling with temperature:
            >>> texts = model.generate_texts(
            ...     ["Explain quantum computing"],
            ...     do_sample=True,
            ...     temperature=0.8,
            ...     max_new_tokens=100
            ... )
            
            Multiple sequences per input:
            >>> texts = model.generate_texts(
            ...     ["Write a haiku about AI"],
            ...     do_sample=True,
            ...     num_return_sequences=3
            ... )
            >>> # texts[0] contains 3 different haikus
            
        Note:
            - Automatically handles batch padding
            - Applies model-specific token fixing if needed
            - Uses default generation parameters unless overridden
            - For token-level access, use the generate() method instead
        """
        # Apply default parameters first, then override with provided args
        default_params = asdict(self.generation_parameters)
        default_params.update(args)
        args = self._validate_args(default_params)

        args["return_dict_in_generate"] = True
        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        sequences = self.generate(**batch, **args).sequences.cpu()
        input_len = batch["input_ids"].shape[1]
        texts = []

        decode_args = {}
        if self.tokenizer.chat_template is not None:
            decode_args["skip_special_tokens"] = True

        for seq in sequences:
            if self.model_type == "CausalLM":
                texts.append(self.tokenizer.decode(seq[input_len:], **decode_args))
            else:
                texts.append(self.tokenizer.decode(seq[1:], **decode_args))

        return texts

    def __call__(self, **args):
        """
        Calls the model on the input batch. Returns the resulted scores.
        """
        return self.model(**args)

    def device(self):
        """
        Returns the device the model is currently loaded on.

        Returns:
            str: device string.
        """
        return self.model.device

    @staticmethod
    def from_pretrained(
        model_path: str,
        generation_params: Optional[Dict] = {},
        add_bos_token: bool = True,
        **kwargs,
    ):
        """
        Create a WhiteboxModel from a pretrained HuggingFace model.
        
        This is the primary factory method for creating WhiteboxModel instances.
        It automatically detects the model type (CausalLM vs Seq2Seq), loads the
        appropriate model and tokenizer, and configures generation parameters.
        
        Parameters:
            model_path: HuggingFace model identifier or local path. Examples:
                - "gpt2", "meta-llama/Llama-2-7b-hf", "google/flan-t5-base"
            generation_params: Default generation parameters as a dict:
                - temperature (float): Sampling temperature
                - max_new_tokens (int): Maximum new tokens to generate
                - top_p (float): Nucleus sampling parameter
                - See GenerationParameters for full list
            add_bos_token: Whether to add beginning-of-sequence token if the
                tokenizer doesn't add it automatically. Default: True
            **kwargs: Additional arguments passed to from_pretrained:
                - device_map (str/dict): Device placement strategy
                - load_in_8bit (bool): Use 8-bit quantization
                - load_in_4bit (bool): Use 4-bit quantization
                - torch_dtype: Override model dtype (e.g., torch.float16)
                - cache_dir (str): Directory for caching models
                - local_files_only (bool): Only use local files
                - revision (str): Model revision/branch to use
                - trust_remote_code (bool): Allow custom model code
                
        Returns:
            WhiteboxModel: Configured model ready for generation and uncertainty estimation
            
        Examples:
            Basic usage:
            >>> model = WhiteboxModel.from_pretrained("gpt2")
            
            With custom device and dtype:
            >>> model = WhiteboxModel.from_pretrained(
            ...     "meta-llama/Llama-2-7b-hf",
            ...     device_map="auto",
            ...     torch_dtype=torch.float16
            ... )
            
            With 8-bit quantization:
            >>> model = WhiteboxModel.from_pretrained(
            ...     "EleutherAI/gpt-j-6B",
            ...     load_in_8bit=True,
            ...     device_map="auto"
            ... )
            
            With custom generation parameters:
            >>> model = WhiteboxModel.from_pretrained(
            ...     "google/flan-t5-large",
            ...     generation_params={
            ...         "temperature": 0.7,
            ...         "max_new_tokens": 200,
            ...         "do_sample": True
            ...     }
            ... )
            
        Note:
            - Automatically detects encoder-decoder vs decoder-only models
            - Sets appropriate padding tokens if missing
            - Configures tokenizer for batch generation
            - For models requiring trust_remote_code, explicitly set it to True
        """
        log.warning(
            "WhiteboxModel#from_pretrained is deprecated and will be removed in the next release. Please instantiate WhiteboxModel directly by passing an already loaded model, tokenizer and model path."
        )

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        generation_params = GenerationParameters(**generation_params)

        if any(["CausalLM" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
        elif any(
            [
                ("Seq2SeqLM" in architecture)
                or ("ConditionalGeneration" in architecture)
                for architecture in config.architectures
            ]
        ):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
            if "falcon" in model_path:
                model.transformer.alibi = True
        elif any(
            ["JAISLMHeadModel" in architecture for architecture in config.architectures]
        ):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs,
            )
        elif any(
            ["BartModel" in architecture for architecture in config.architectures]
        ):
            model_type = "Seq2SeqLM"
            model = BartForConditionalGeneration.from_pretrained(model_path, **kwargs)
        else:
            raise ValueError(
                f"Model {model_path} is not adapted for the sequence generation task"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=add_bos_token,
            **kwargs,
        )

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        instance = WhiteboxModel(
            model, tokenizer, model_path, model_type, generation_params
        )

        return instance

    def tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input texts batch into a dictionary using the model tokenizer.

        Parameters:
            texts (List[str]): list of input texts batch.
        Returns:
            dict[str, torch.Tensor]: tensors dictionary obtained by tokenizing input texts batch.
        """
        # Apply chat template if tokenizer has it
        add_start_symbol = True
        if self.tokenizer.chat_template is not None:
            formatted_texts = []
            for chat in texts:
                if isinstance(chat, str):
                    chat = [{"role": "user", "content": chat}]
                formatted_chat = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                formatted_texts.append(formatted_chat)
            texts = formatted_texts

            add_start_symbol = False

        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=add_start_symbol,
        )


def create_ensemble(
    models: List[WhiteboxModel] = [],
    mc: bool = False,
    seed: int = 1,
    mc_seeds: List[int] = [1],
    ensembling_mode: str = "pe",
    dropout_rate: float = 0.1,
    **kwargs,
) -> WhiteboxModel:
    model = models[0]
    ens = model.model

    ens.__class__ = type(
        "EnsembleModel", (model.model.__class__, EnsembleGenerationMixin), {}
    )

    if mc:
        ens.mc = True
        ens.mc_seeds = mc_seeds
        ens.base_seed = seed
        ens.ensembling_mode = ensembling_mode
        ens.mc_models_num = len(mc_seeds)
        ens.mc_seeds = mc_seeds

        replace_dropout(
            ens.config._name_or_path, ens, p=dropout_rate, share_across_tokens=True
        )
        ens.train()
    else:
        raise ValueError(
            "Only Monte-Carlo ensembling is available. Please set the corresponding argument value to True"
        )

    return model
