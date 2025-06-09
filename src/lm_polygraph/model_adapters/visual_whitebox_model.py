"""
Visual language model adapter for uncertainty estimation with multimodal inputs.

This module provides support for vision-language models (VLMs) in the LM-Polygraph
framework, enabling uncertainty estimation for models that process both text and
image inputs. It extends the whitebox model interface to handle multimodal generation
tasks while maintaining compatibility with uncertainty estimators.
"""

import torch
import requests
from lm_polygraph.utils.model import Model
from PIL import Image
from typing import List, Optional, Dict, Union
from dataclasses import asdict
import logging

from lm_polygraph.utils.generation_parameters import GenerationParameters
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizer,
    GenerationConfig,
)


log = logging.getLogger("lm_polygraph")


class VisualWhiteboxModel(Model):
    """
    Whitebox model adapter for vision-language models with uncertainty estimation.
    
    This class enables uncertainty quantification for multimodal models that process
    both images and text. It provides full access to model internals (logits, attention)
    while handling the complexities of multimodal input processing.
    
    Supported model types include:
    - BLIP-2 (Salesforce/blip2-*)
    - LLaVA (llava-hf/*)
    - Flamingo-style models
    - Other HuggingFace Vision2Seq models
    
    Key features:
    - Combined image-text input processing
    - Access to token probabilities for uncertainty estimation
    - Support for multiple images per batch
    - Custom stopping criteria for generation
    - Compatible with all whitebox uncertainty methods
    
    Attributes:
        model: The underlying Vision2Seq model
        processor_visual: Multimodal processor for images and text
        tokenizer: Text tokenizer (extracted from processor)
        model_path: Model identifier or path
        model_type: Set to "VisualLM"
        images: Loaded PIL images for generation
        generation_parameters: Default generation settings
        
    Examples:
        Basic usage with image URL:
        >>> from lm_polygraph.model_adapters import VisualWhiteboxModel
        >>> model = VisualWhiteboxModel.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b",
        ...     model_type="VisualLM",
        ...     image_urls=["https://example.com/image.jpg"]
        ... )
        >>> 
        >>> # Generate description with uncertainty
        >>> from lm_polygraph.estimators import TokenEntropy
        >>> estimator = TokenEntropy()
        >>> result = estimate_uncertainty(
        ...     model, estimator,
        ...     "Describe this image in detail:"
        ... )
        
        With local images:
        >>> model = VisualWhiteboxModel.from_pretrained(
        ...     "llava-hf/llava-1.5-7b-hf",
        ...     model_type="VisualLM",
        ...     image_paths=["path/to/image1.jpg", "path/to/image2.jpg"]
        ... )
        >>> 
        >>> # Ask questions about images
        >>> texts = model.generate_texts([
        ...     "What objects are in this image?",
        ...     "What is the main color?"
        ... ])
        
        Direct initialization:
        >>> from transformers import AutoModelForVision2Seq, AutoProcessor
        >>> base_model = AutoModelForVision2Seq.from_pretrained("model-name")
        >>> processor = AutoProcessor.from_pretrained("model-name")
        >>> model = VisualWhiteboxModel(
        ...     base_model, processor,
        ...     image_paths=["image.jpg"]
        ... )
        
    See Also:
        WhiteboxModel: Standard text-only model adapter
        BlackboxModel: For API-based multimodal models
        
    Note:
        - Images are loaded once during initialization for efficiency
        - All text inputs in a batch use the same set of images
        - Different VLMs may have different prompt formats
    """

    def __init__(
        self,
        model: AutoModelForVision2Seq,
        processor_visual: AutoProcessor,
        model_path: str = None,
        model_type: str = "VisualLM",
        image_urls: list = None,
        image_paths: list = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
    ):
        """
        Initialize visual whitebox model for multimodal uncertainty estimation.
        
        Parameters:
            model: Pre-loaded HuggingFace Vision2Seq model
            processor_visual: Corresponding multimodal processor that handles
                both image and text inputs
            model_path: Model identifier or path for tracking
            model_type: Type identifier, typically "VisualLM"
            image_urls: List of image URLs to download and use. Mutually
                exclusive with image_paths
            image_paths: List of local image file paths. Mutually exclusive
                with image_urls
            generation_parameters: Default generation settings
            
        Raises:
            ValueError: If neither image_urls nor image_paths is provided
            
        Note:
            Images are downloaded (if URLs) or loaded immediately during
            initialization. For dynamic image loading, create new model
            instances or modify the images attribute directly.
        """
        super().__init__(model_path, model_type)
        self.model = model
        self.model_type = model_type
        self.processor_visual = processor_visual
        self.tokenizer = self.processor_visual.tokenizer
        self.generation_parameters = generation_parameters
        if image_urls:
            self.images = [
                Image.open(requests.get(img_url, stream=True).raw)
                for img_url in image_urls
            ]
        elif image_paths:
            self.images = [Image.open(img_path) for img_path in image_paths]
        else:
            raise ValueError("Either image_path or image_url must be provided")
        self.generation_parameters = generation_parameters or GenerationParameters()

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
        keys_to_remove = [
            "presence_penalty",
            "generate_until",
            "allow_newlines",
            "return_dict",
        ]
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
            self.tokenizer = self.processor_visual.tokenizer

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
        Generate model outputs with scores for multimodal inputs.
        
        Processes combined image-text inputs through the vision-language model,
        returning detailed generation outputs including token probabilities for
        uncertainty estimation. Handles proper formatting of multimodal inputs
        and preserves original probability scores.
        
        Parameters:
            **args: Generation arguments including:
                - input_ids: Tokenized text input sequences
                - pixel_values: Processed image tensors (if required by model)
                - attention_mask: Attention mask for text inputs
                - max_new_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling
                - output_scores: Return token probabilities (default: True)
                - stopping_criteria: Custom stopping conditions
                - Other HuggingFace generation parameters
                
        Returns:
            ModelOutput: Generation output containing:
                - sequences: Generated token IDs including input
                - scores: Original token log probabilities (not modified)
                - generation_scores: Modified scores (if any processing applied)
                - attentions: Attention weights (if requested)
                
        Examples:
            >>> # Tokenize with images
            >>> inputs = model.processor_visual(
            ...     text=["What do you see?"],
            ...     images=model.images,
            ...     return_tensors="pt"
            ... )
            >>> outputs = model.generate(**inputs, max_new_tokens=50)
            >>> # outputs.scores contains probabilities for uncertainty
            
        Note:
            The method preserves original token scores before any generation
            parameter modifications, which is crucial for accurate uncertainty
            estimation.
        """
        default_params = asdict(self.generation_parameters)
        args.pop("return_dict", None)
        if "input_ids" in args and len(self.generation_parameters.generate_until) > 0:
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
        if "generation_config" not in args:
            generation_config = GenerationConfig(
                **{
                    k: v
                    for k, v in args.items()
                    if k in GenerationConfig.__annotations__
                }
            )
            # Remove generation parameters that are now in the config
            args = {
                k: v
                for k, v in args.items()
                if k not in GenerationConfig.__annotations__
            }
            args["generation_config"] = generation_config

        # Ensure we're not passing return_dict at all
        args.pop("return_dict", None)
        if "generation_config" in args:
            args["generation_config"].return_dict_in_generate = True

        generation = self.model.generate(**args)

        if hasattr(generation, "scores"):
            generation.generation_scores = generation.scores
            generation.scores = processor.scores

        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generate text completions for prompts with associated images.
        
        High-level method that handles the complete pipeline of multimodal
        generation: processes text prompts with pre-loaded images, generates
        responses, and decodes them to readable text.
        
        Parameters:
            input_texts: List of text prompts/questions about the images.
                Each prompt is paired with the same set of images loaded
                during model initialization
            **args: Additional generation parameters:
                - max_new_tokens: Maximum tokens to generate per prompt
                - temperature: Sampling temperature (0.0 = greedy)
                - do_sample: Whether to use sampling
                - num_beams: Beam search width
                - top_p: Nucleus sampling threshold
                - repetition_penalty: Penalty for repeated tokens
                - Any other HuggingFace generation parameters
                
        Returns:
            List[str]: Generated text responses for each input prompt.
                Length matches input_texts length.
                
        Examples:
            Single prompt:
            >>> texts = model.generate_texts(
            ...     ["What is happening in this image?"],
            ...     max_new_tokens=100
            ... )
            >>> print(texts[0])
            
            Multiple prompts about same images:
            >>> prompts = [
            ...     "Describe the scene",
            ...     "What colors are dominant?",
            ...     "Is this indoors or outdoors?"
            ... ]
            >>> responses = model.generate_texts(prompts, temperature=0.7)
            
            With beam search:
            >>> texts = model.generate_texts(
            ...     ["Generate a detailed caption:"],
            ...     num_beams=4,
            ...     max_new_tokens=150
            ... )
            
        Note:
            - All prompts use the same images loaded during initialization
            - To use different images, create new model instances
            - Some VLMs require specific prompt formats (e.g., "Question: ... Answer:")
            - The method handles special tokens based on model's chat template
        """
        args = self._validate_args(args)
        batch: Dict[str, torch.Tensor] = self.processor_visual(
            text=input_texts,
            images=self.images,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        args.pop("return_dict", None)
        sequences = self.generate(**batch, **args).sequences.cpu()
        input_len = batch["input_ids"].shape[1]
        texts = []

        decode_args = {}
        if self.tokenizer.chat_template is not None:
            decode_args["skip_special_tokens"] = True

        for seq in sequences:
            texts.append(self.processor_visual.decode(seq[input_len:], **decode_args))
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
        model_type: str,
        image_urls: list = None,
        image_paths: list = None,
        generation_params: Optional[Dict] = {},
        add_bos_token: bool = True,
        **kwargs,
    ):
        """
        Create a VisualWhiteboxModel from a pretrained vision-language model.
        
        Factory method that loads a vision-language model from HuggingFace hub
        or local path, along with its processor, and prepares it for uncertainty
        estimation with the specified images.
        
        Parameters:
            model_path: HuggingFace model ID or local path to model.
                Examples: "Salesforce/blip2-opt-2.7b", "llava-hf/llava-1.5-7b-hf"
            model_type: Type identifier, typically "VisualLM" for consistency
            image_urls: List of image URLs to download and use. Mutually
                exclusive with image_paths
            image_paths: List of local image file paths. Mutually exclusive
                with image_urls
            generation_params: Dictionary of default generation parameters:
                - temperature: Sampling temperature
                - max_new_tokens: Maximum generation length
                - top_p: Nucleus sampling threshold
                See GenerationParameters for full list
            add_bos_token: Whether to add beginning-of-sequence token to
                tokenizer configuration. Default: True
            **kwargs: Additional arguments for model loading:
                - device_map: Device placement ("auto", "cuda:0", etc.)
                - torch_dtype: Model precision (torch.float16, etc.)
                - load_in_8bit: Enable 8-bit quantization
                - load_in_4bit: Enable 4-bit quantization
                - cache_dir: Directory for model caching
                - revision: Model revision to load
                - trust_remote_code: Allow custom model code
                
        Returns:
            VisualWhiteboxModel: Initialized model ready for generation
            
        Examples:
            Basic usage:
            >>> model = VisualWhiteboxModel.from_pretrained(
            ...     "Salesforce/blip2-opt-2.7b",
            ...     "VisualLM",
            ...     image_urls=["https://example.com/cat.jpg"]
            ... )
            
            With local images and GPU:
            >>> model = VisualWhiteboxModel.from_pretrained(
            ...     "llava-hf/llava-1.5-7b-hf",
            ...     "VisualLM",
            ...     image_paths=["img1.jpg", "img2.jpg"],
            ...     device_map="auto",
            ...     torch_dtype=torch.float16
            ... )
            
            With quantization:
            >>> model = VisualWhiteboxModel.from_pretrained(
            ...     "Salesforce/blip2-flan-t5-xxl",
            ...     "VisualLM",
            ...     image_paths=["image.png"],
            ...     load_in_8bit=True,
            ...     generation_params={"temperature": 0.9}
            ... )
            
        Raises:
            ValueError: If neither image_urls nor image_paths provided
            
        Note:
            This method is marked as deprecated in favor of direct initialization
            with pre-loaded models, but remains for backward compatibility.
        """
        log.warning(
            "WhiteboxModel#from_pretrained is deprecated and will be removed in the next release. Please instantiate WhiteboxModel directly by passing an already loaded model, tokenizer and model path."
        )

        generation_params = GenerationParameters(**generation_params)
        model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)

        processor_visual = AutoProcessor.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=add_bos_token,
            **kwargs,
        )

        model.eval()
        if processor_visual.tokenizer.pad_token is None:
            processor_visual.tokenizer.pad_token = processor_visual.tokenizer.eos_token
        instance = VisualWhiteboxModel(
            model,
            processor_visual,
            model_path,
            model_type,
            image_urls,
            image_paths,
            generation_params,
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

        return self.tokenizer(texts, padding=True, return_tensors="pt")
