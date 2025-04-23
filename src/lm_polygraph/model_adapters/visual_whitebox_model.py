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
    White-box model class. Have access to model scores and logits. Currently implemented only for Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import VisualWhiteboxModel
    ... )
    ```
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
        Parameters:
            model (AutoModelForCausalLM): HuggingFace model.
            tokenizer (AutoTokenizer): HuggingFace tokenizer.
            model_path (Optional[str]): Unique model path in HuggingFace.
            model_type (str): Additional model specifications.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
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
        Generates the model output with scores from batch formed by HF Tokenizer.

        Parameters:
            **args: Any arguments that can be passed to model.generate function from HuggingFace.
        Returns:
            ModelOutput: HuggingFace generation output with scores overriden with original probabilities.
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
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
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
        image_urls: list,
        image_paths: list,
        generation_params: Optional[Dict] = {},
        add_bos_token: bool = True,
        **kwargs,
    ):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            generation_params (Dict): generation arguments for
                lm_polygraph.utils.generation_parametersGenerationParameters
            add_bos_token (bool): tokenizer argument. Default: True.
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
