import torch
import openai
import time
import logging
import json
import sys

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
)
from huggingface_hub import InferenceClient

from lm_polygraph.utils.generation_parameters import (
    GenerationParameters,
    GenerationParametersFactory,
)
from lm_polygraph.utils.ensemble_utils.ensemble_generator import EnsembleGenerationMixin
from lm_polygraph.utils.ensemble_utils.dropout import replace_dropout
from lm_polygraph.utils.tools import ToolManager
from lm_polygraph.utils.tool_calling import execute_tool_calling_workflow

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
    Black-box model class. Have no access to model scores and logits.
    Currently implemented blackbox models: OpenAI models, Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> model = BlackboxModel.from_openai(
    ...     'YOUR_OPENAI_TOKEN',
    ...     'gpt-3.5-turbo'
    ... )
    ```

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> model = BlackboxModel.from_huggingface(
    ...     hf_api_token='YOUR_API_TOKEN',
    ...     hf_model_id='google/t5-large-ssm-nqo'
    ... )
    ```
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model_path: str = None,
        hf_api_token: str = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
        supports_logprobs: bool = False,
        tool_manager: Optional[ToolManager] = None,
        tool_mandatory: bool = False,
        tool_name: Optional[str] = None,
    ):
        """
        Parameters:
            openai_api_key (Optional[str]): OpenAI API key if the blackbox model comes from OpenAI. Default: None.
            model_path (Optional[str]): Unique model path. Openai model name, if `openai_api_key` is specified,
                huggingface path, if `hf_api_token` is specified. Default: None.
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            generation_parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
            supports_logprobs (bool): Whether the model supports returning log probabilities. Default: False.
            tool_manager (Optional[ToolManager]): Tool manager with available tools. Default: None.
            tool_mandatory (bool): Whether tool usage is mandatory. Default: False.
            tool_name (Optional[str]): Name of the tool to use (required if mandatory). Default: None.
        """
        super().__init__(model_path, "Blackbox")
        self.generation_parameters = generation_parameters
        self.openai_api_key = openai_api_key
        self.supports_logprobs = supports_logprobs
        self.tool_manager = tool_manager
        self.tool_mandatory = tool_mandatory
        self.tool_name = tool_name
        self._tool_usage_info = []  # Track tool usage for logging

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
            "allow_newlines",
            "stop_strings",
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
        client = InferenceClient(model=self.model_path, token=self.hf_api_token)
        response = client.chat_completion(payload)
        raw_json = json.dumps(response, indent=2)
        return raw_json

    @staticmethod
    def from_huggingface(hf_api_token: str, hf_model_id: str, **kwargs):
        """
        Initializes a blackbox model from huggingface.

        Parameters:
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            hf_model_id (Optional[str]): model path in huggingface.
            **kwargs: Additional arguments including tool_manager, tool_mandatory, tool_name.
        """
        generation_parameters = kwargs.pop(
            "generation_parameters", GenerationParameters()
        )
        tool_manager = kwargs.pop("tool_manager", None)
        tool_mandatory = kwargs.pop("tool_mandatory", False)
        tool_name = kwargs.pop("tool_name", None)
        return BlackboxModel(
            hf_api_token=hf_api_token,
            model_path=hf_model_id,
            generation_parameters=generation_parameters,
            tool_manager=tool_manager,
            tool_mandatory=tool_mandatory,
            tool_name=tool_name,
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
            **kwargs: Additional arguments including tool_manager, tool_mandatory, tool_name.
        """
        generation_parameters = kwargs.pop(
            "generation_parameters", GenerationParameters()
        )
        tool_manager = kwargs.pop("tool_manager", None)
        tool_mandatory = kwargs.pop("tool_mandatory", False)
        tool_name = kwargs.pop("tool_name", None)
        return BlackboxModel(
            openai_api_key=openai_api_key,
            model_path=model_path,
            supports_logprobs=supports_logprobs,
            generation_parameters=generation_parameters,
            tool_manager=tool_manager,
            tool_mandatory=tool_mandatory,
            tool_name=tool_name,
        )

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        # Check if tool calling is enabled (backward compatible - only if tools are configured)
        # Use get() first to check, then pop() only if needed to avoid modifying args unnecessarily
        use_tools = args.get("use_tools", None)
        if use_tools is None:
            # Default: use tools only if tool_manager is set and has tools
            # When tool_manager is None (no tools configured), this will be False
            use_tools = self.tool_manager is not None and self.tool_manager.has_tools()
        else:
            # Remove use_tools from args if it was explicitly provided
            args.pop("use_tools")
        
        # Only use tool calling workflow if tools are configured and enabled
        if use_tools and self.tool_manager is not None and self.tool_manager.has_tools():
            # Use tool calling workflow
            results = []
            tool_names_used = []
            tools_used = []
            max_new_tokens = args.get("max_new_tokens", self.generation_parameters.max_new_tokens)
            for question in input_texts:
                result, tool_name_used, tool_was_used = execute_tool_calling_workflow(
                    model=self,
                    question=question,
                    tool_manager=self.tool_manager,
                    mandatory=self.tool_mandatory,
                    tool_name=self.tool_name,
                    max_new_tokens=max_new_tokens,
                    **{k: v for k, v in args.items() if k != "max_new_tokens"}
                )
                results.append(result)
                tool_names_used.append(tool_name_used)
                tools_used.append(tool_was_used)
            
            # Store tool usage info for logging
            if not hasattr(self, '_tool_usage_info'):
                self._tool_usage_info = []
            self._tool_usage_info.append({
                'tool_names': tool_names_used,
                'tools_used': tools_used,
                'input_texts': input_texts
            })
            
            # Log tool usage
            for i, (tool_name, tool_used) in enumerate(zip(tool_names_used, tools_used)):
                if tool_used and tool_name:
                    log.info(f"Tool used for input {i}: {tool_name}")
                elif not tool_used:
                    log.info(f"No tool used for input {i}")
            
            return results
        
        # Normal generation path (backward compatible - no tools)
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
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    output = self._query(messages)

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
    White-box model class. Have access to model scores and logits. Currently implemented only for Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import WhiteboxModel
    >>> model = WhiteboxModel.from_pretrained(
    ...     "bigscience/bloomz-3b",
    ... )
    ```
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_path: str = None,
        model_type: str = "CausalLM",
        generation_parameters: GenerationParameters = GenerationParameters(),
        instruct: bool = False,
        tool_manager: Optional[ToolManager] = None,
        tool_mandatory: bool = False,
        tool_name: Optional[str] = None,
    ):
        """
        Parameters:
            model (AutoModelForCausalLM): HuggingFace model.
            tokenizer (AutoTokenizer): HuggingFace tokenizer.
            model_path (Optional[str]): Unique model path in HuggingFace.
            model_type (str): Additional model specifications.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
            tool_manager (Optional[ToolManager]): Tool manager with available tools. Default: None.
            tool_mandatory (bool): Whether tool usage is mandatory. Default: False.
            tool_name (Optional[str]): Name of the tool to use (required if mandatory). Default: None.
        """
        super().__init__(model_path, model_type)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_parameters = generation_parameters
        self.instruct = instruct
        self.tool_manager = tool_manager
        self.tool_mandatory = tool_mandatory
        self.tool_name = tool_name
        self._tool_usage_info = []  # Track tool usage for logging

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
        keys_to_remove = ["presence_penalty", "allow_newlines"]
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

    def generate(self, **args):
        """
        Generates the model output with scores from batch formed by HF Tokenizer.

        Parameters:
            **args: Any arguments that can be passed to model.generate function from HuggingFace.
        Returns:
            ModelOutput: HuggingFace generation output with scores overriden with original probabilities.
        """
        print(f"[DEBUG generate] Called with args keys: {list(args.keys())}", file=sys.stderr, flush=True)
        default_params = asdict(self.generation_parameters)
        print(f"[DEBUG generate] Default params keys: {list(default_params.keys())}", file=sys.stderr, flush=True)

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

        if "stop_strings" in args:
            args["tokenizer"] = self.tokenizer

        args = self._validate_args(args)
        print(f"[DEBUG generate] About to call self.model.generate() with max_new_tokens={args.get('max_new_tokens', 'not set')}", file=sys.stderr, flush=True)
        print(f"[DEBUG generate] Input shape: {args.get('input_ids', 'N/A').shape if hasattr(args.get('input_ids', None), 'shape') else 'N/A'}", file=sys.stderr, flush=True)
        import time
        gen_start = time.time()
        generation = self.model.generate(**args)
        gen_elapsed = time.time() - gen_start
        print(f"[DEBUG generate] Model.generate() completed in {gen_elapsed:.2f} seconds", file=sys.stderr, flush=True)

        # override generation.scores with original scores from model
        generation.generation_scores = generation.scores
        generation.scores = processor.scores
        print(f"[DEBUG generate] Returning generation output", file=sys.stderr, flush=True)

        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        import sys
        # DEBUG: Log tool_manager status at the start
        print(f"[DEBUG WhiteboxModel.generate_texts] ENTRY: input_texts count={len(input_texts)}, "
              f"tool_manager={self.tool_manager is not None}, "
              f"has_tools={self.tool_manager.has_tools() if self.tool_manager else False}, "
              f"tool_mandatory={self.tool_mandatory}, tool_name={self.tool_name}", 
              file=sys.stderr, flush=True)
        
        # Check if tool calling is enabled (backward compatible - only if tools are configured)
        # Use get() first to check, then pop() only if needed to avoid modifying args unnecessarily
        use_tools = args.get("use_tools", None)
        print(f"[DEBUG WhiteboxModel.generate_texts] use_tools from args: {use_tools}", file=sys.stderr, flush=True)
        
        if use_tools is None:
            # Default: use tools only if tool_manager is set and has tools
            # When tool_manager is None (no tools configured), this will be False
            use_tools = self.tool_manager is not None and self.tool_manager.has_tools()
            print(f"[DEBUG WhiteboxModel.generate_texts] use_tools computed from tool_manager: {use_tools}", file=sys.stderr, flush=True)
        else:
            # Remove use_tools from args if it was explicitly provided
            args.pop("use_tools")
            print(f"[DEBUG WhiteboxModel.generate_texts] use_tools was explicitly set to: {use_tools}", file=sys.stderr, flush=True)
        
        # Only use tool calling workflow if tools are configured and enabled
        if use_tools and self.tool_manager is not None and self.tool_manager.has_tools():
            print(f"[DEBUG WhiteboxModel.generate_texts] ===== USING TOOL CALLING WORKFLOW =====", file=sys.stderr, flush=True)
            print(f"[DEBUG WhiteboxModel.generate_texts] tool_manager.tools: {[t.get_name() for t in self.tool_manager.tools]}", file=sys.stderr, flush=True)
            
            # Use tool calling workflow
            results = []
            tool_names_used = []
            tools_used = []
            max_new_tokens = args.get("max_new_tokens", self.generation_parameters.max_new_tokens)
            for question in input_texts:
                result, tool_name_used, tool_was_used = execute_tool_calling_workflow(
                    model=self,
                    question=question,
                    tool_manager=self.tool_manager,
                    mandatory=self.tool_mandatory,
                    tool_name=self.tool_name,
                    max_new_tokens=max_new_tokens,
                    **{k: v for k, v in args.items() if k != "max_new_tokens"}
                )
                results.append(result)
                tool_names_used.append(tool_name_used)
                tools_used.append(tool_was_used)
            
            # Store tool usage info for logging
            if not hasattr(self, '_tool_usage_info'):
                self._tool_usage_info = []
            self._tool_usage_info.append({
                'tool_names': tool_names_used,
                'tools_used': tools_used,
                'input_texts': input_texts
            })
            
            # Log tool usage
            for i, (tool_name, tool_used) in enumerate(zip(tool_names_used, tools_used)):
                if tool_used and tool_name:
                    log.info(f"Tool used for input {i}: {tool_name}")
                elif not tool_used:
                    log.info(f"No tool used for input {i}")
            
            return results
        else:
            print(f"[DEBUG WhiteboxModel.generate_texts] ===== SKIPPING TOOL CALLING =====", file=sys.stderr, flush=True)
            print(f"[DEBUG WhiteboxModel.generate_texts] Reason: use_tools={use_tools}, "
                  f"tool_manager={self.tool_manager is not None}, "
                  f"has_tools={self.tool_manager.has_tools() if self.tool_manager else False}", 
                  file=sys.stderr, flush=True)
            print(f"[DEBUG WhiteboxModel.generate_texts] Continuing with normal generation path for {len(input_texts)} inputs", file=sys.stderr, flush=True)
        
        # Normal generation path (backward compatible - no tools)
        # Apply default parameters first, then override with provided args
        print(f"[DEBUG generate_texts] Step 1: Getting default params", file=sys.stderr, flush=True)
        default_params = asdict(self.generation_parameters)
        print(f"[DEBUG generate_texts] Step 2: Updating with args, args keys: {list(args.keys())}", file=sys.stderr, flush=True)
        default_params.update(args)
        print(f"[DEBUG generate_texts] Step 3: Validating args", file=sys.stderr, flush=True)
        args = self._validate_args(default_params)
        print(f"[DEBUG generate_texts] Step 4: Setting return_dict_in_generate=True", file=sys.stderr, flush=True)

        args["return_dict_in_generate"] = True
        print(f"[DEBUG generate_texts] Step 5: Tokenizing {len(input_texts)} inputs", file=sys.stderr, flush=True)
        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        print(f"[DEBUG generate_texts] Step 6: Moving batch to device {self.device()}", file=sys.stderr, flush=True)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        print(f"[DEBUG generate_texts] Step 7: Calling model.generate() with batch shape {batch['input_ids'].shape}", file=sys.stderr, flush=True)
        print(f"[DEBUG generate_texts] Generation args: max_new_tokens={args.get('max_new_tokens', 'not set')}, other keys: {[k for k in args.keys() if k not in ['input_ids', 'attention_mask', 'return_dict_in_generate']]}", file=sys.stderr, flush=True)
        print(f"[DEBUG generate_texts] About to call self.generate() - this may take a while...", file=sys.stderr, flush=True)
        try:
            import time
            start_time = time.time()
            generation_output = self.generate(**batch, **args)
            elapsed_time = time.time() - start_time
            print(f"[DEBUG generate_texts] Step 8: Generation completed in {elapsed_time:.2f} seconds, getting sequences", file=sys.stderr, flush=True)
            sequences = generation_output.sequences.cpu()
            print(f"[DEBUG generate_texts] Step 9: Sequences shape: {sequences.shape}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[DEBUG generate_texts] ERROR in generation: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
        input_len = batch["input_ids"].shape[1]
        texts = []

        decode_args = {}
        if self.tokenizer.chat_template is not None:
            decode_args["skip_special_tokens"] = True

        print(f"[DEBUG generate_texts] Generated {len(sequences)} sequences, decoding...", file=sys.stderr, flush=True)
        for i, seq in enumerate(sequences):
            if self.model_type == "CausalLM":
                decoded = self.tokenizer.decode(seq[input_len:], **decode_args)
                texts.append(decoded)
                if i == 0:
                    print(f"[DEBUG generate_texts] First decoded text (first 200 chars): {decoded[:200]}", file=sys.stderr, flush=True)
            else:
                decoded = self.tokenizer.decode(seq[1:], **decode_args)
                texts.append(decoded)
                if i == 0:
                    print(f"[DEBUG generate_texts] First decoded text (first 200 chars): {decoded[:200]}", file=sys.stderr, flush=True)

        print(f"[DEBUG generate_texts] Returning {len(texts)} decoded texts", file=sys.stderr, flush=True)
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
        instruct: bool = False,
        tool_manager: Optional[ToolManager] = None,
        tool_mandatory: bool = False,
        tool_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            generation_params (Dict): generation arguments for
                lm_polygraph.utils.generation_parametersGenerationParameters
            add_bos_token (bool): tokenizer argument. Default: True.
            instruct (bool): Whether the model is instruction-tuned. Default: False.
            tool_manager (Optional[ToolManager]): Tool manager with available tools. Default: None.
            tool_mandatory (bool): Whether tool usage is mandatory. Default: False.
            tool_name (Optional[str]): Name of the tool to use (required if mandatory). Default: None.
        """
        log.warning(
            "WhiteboxModel#from_pretrained is deprecated and will be removed in the next release. Please instantiate WhiteboxModel directly by passing an already loaded model, tokenizer and model path."
        )

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

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

        generation_params = GenerationParametersFactory.from_params(
            yaml_config=generation_params,
            native_config=asdict(model.config),
        )

        instance = WhiteboxModel(
            model, tokenizer, model_path, model_type, generation_params,
            instruct=instruct,
            tool_manager=tool_manager,
            tool_mandatory=tool_mandatory,
            tool_name=tool_name,
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
        if self.instruct:
            # Check if tokenizer has a chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                formatted_texts = []
                for chat in texts:
                    if isinstance(chat, str):
                        chat = [{"role": "user", "content": chat}]
                    try:
                        formatted_chat = self.tokenizer.apply_chat_template(
                            chat, add_generation_prompt=True, tokenize=False
                        )
                        formatted_texts.append(formatted_chat)
                    except (ValueError, TypeError) as e:
                        # If chat template fails (e.g., doesn't support tools), fall back to plain text
                        log.warning(f"Chat template application failed: {e}. Falling back to plain text formatting.")
                        formatted_texts.append(chat[0]["content"] if isinstance(chat, list) and len(chat) > 0 else str(chat))
                texts = formatted_texts
                add_start_symbol = False
            else:
                # No chat template available, but instruct=True was set
                # This can happen with base models - log a warning and proceed with plain text
                log.warning("instruct=True but tokenizer has no chat_template. Using plain text formatting.")
                add_start_symbol = True
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
