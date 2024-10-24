import requests
import torch
import sys
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
        parameters: GenerationParameters = GenerationParameters(),
    ):
        """
        Parameters:
            openai_api_key (Optional[str]): OpenAI API key if the blackbox model comes from OpenAI. Default: None.
            model_path (Optional[str]): Unique model path. Openai model name, if `openai_api_key` is specified,
                huggingface path, if `hf_api_token` is specified. Default: None.
            hf_api_token (Optional[str]): Huggingface API token if the blackbox model comes from HF. Default: None.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
        """
        super().__init__(model_path, "Blackbox")
        self.parameters = parameters
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.hf_api_token = hf_api_token

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
        return BlackboxModel(hf_api_token=hf_api_token, model_path=hf_model_id)

    @staticmethod
    def from_openai(openai_api_key: str, model_path: str, **kwargs):
        """
        Initializes a blackbox model from OpenAI API.

        Parameters:
            openai_api_key (Optional[str]): OpenAI API key. Default: None.
            model_path (Optional[str]): model name in OpenAI.
        """
        return BlackboxModel(openai_api_key=openai_api_key, model_path=model_path)

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        if any(
            args.get(arg, False)
            for arg in ["output_scores", "output_attentions", "output_hidden_states"]
        ):
            raise Exception("Cannot access logits for blackbox model")

        for delete_key in [
            "do_sample",
            "min_length",
            "top_k",
            "repetition_penalty",
            "min_new_tokens",
        ]:
            args.pop(delete_key, None)
        for key, replace_key in [
            ("num_return_sequences", "n"),
            ("max_length", "max_tokens"),
            ("max_new_tokens", "max_tokens"),
        ]:
            if key in args.keys():
                args[replace_key] = args[key]
                args.pop(key)
        texts = []

        if self.openai_api_key is not None:
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
                        response = openai.ChatCompletion.create(
                            model=self.model_path,
                            messages=messages,
                            **args,
                        )
                        break
                    except Exception as e:
                        if retries > 4:
                            raise Exception from e
                        else:
                            retries += 1
                            continue

                if args["n"] == 1:
                    texts.append(response.choices[0].message.content)
                else:
                    texts.append([resp.message.content for resp in response.choices])
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
                            print(f"{output['error']}")
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
        Not implemented for blackbox models.
        """
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


def _validate_args(args):
    if "presence_penalty" in args.keys() and args["presence_penalty"] != 0.0:
        sys.stderr.write(
            "Skipping requested argument presence_penalty={}".format(
                args["presence_penalty"]
            )
        )

    # remove arguments that are not supported by the HF model.generate function
    keys_to_remove = ["presence_penalty", "generate_until", "allow_newlines"]
    for key in keys_to_remove:
        args.pop(key, None)

    return args


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
        Generates the model output with scores from batch formed by HF Tokenizer.

        Parameters:
            **args: Any arguments that can be passed to model.generate function from HuggingFace.
        Returns:
            ModelOutput: HuggingFace generation output with scores overriden with original probabilities.
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
        args = _validate_args(args)

        generation = self.model.generate(**args)

        # override generation.scores with original scores from model
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
        args = _validate_args(args)
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
