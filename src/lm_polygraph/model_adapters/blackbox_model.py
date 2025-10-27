from dataclasses import asdict
from typing import List

from lm_polygraph.utils.model import Model
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.model_adapters.api_provider_adapter import get_adapter


class BlackboxModel(Model):
    """
    Black-box model class. Have no access to model scores and logits.
    Currently implemented blackbox models: OpenAI models, Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> model = BlackboxModel(
    ...     model_path='gpt-3.5-turbo',
    ...     api_provider_name='openai'
    ... )
    ```

    ```python
    >>> from lm_polygraph import BlackboxModel
    >>> model = BlackboxModel(
    ...     model_path='google/t5-large-ssm-nqo',
    ...     api_provider_name='huggingface'
    ... )
    ```
    """

    def __init__(
        self,
        model_path: str = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
        api_provider_name: str = "openai",
    ):
        """
        Parameters:
            model_path (Optional[str]): Unique model path or identifier understood by the provider.
            generation_parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
        """
        super().__init__(model_path, "Blackbox")
        self.generation_parameters = generation_parameters
        self.api_provider_name = api_provider_name

        # Initialize the adapter for this provider
        self.adapter = get_adapter(self.api_provider_name)

    @property
    def supports_logprobs(self) -> bool:
        """Expose adapter-defined logprob support."""
        return self.adapter.supports_logprobs(self.model_path)

    def _validate_args(self, args):
        """
        Validates and adapts arguments for BlackboxModel generation using the configured adapter.

        Parameters:
            args (dict): The arguments to validate.

        Returns:
            dict: Validated and adapted arguments.
        """
        # Use the adapter to validate parameter ranges first
        validated_args = self.adapter.validate_parameter_ranges(args)

        # Then adapt the request format for the specific provider
        return self.adapter.adapt_request(validated_args)

    def prepare_input(self, prompt: List[str]) -> List[dict]:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(
            isinstance(item, dict) for item in prompt
        ):
            messages = prompt
        else:
            raise ValueError(
                "Invalid prompt format. Must be either a string or a list of dictionaries."
            )

        return messages

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        # Apply default parameters first, then override with provided args
        default_params = asdict(self.generation_parameters)
        default_params.update(args)
        args = self._validate_args(default_params)

        # Check if we're trying to access features that require logprobs support
        requires_logprobs = getattr(args, "output_scores", False)

        # Use adapter to check logprobs support (considering model-specific rules)
        if requires_logprobs and not self.supports_logprobs:
            raise Exception(
                f"Cannot access logits for blackbox model with provider '{self.api_provider_name}'"
            )

        return self.adapter.generate_texts(self, input_texts, args)

    def generate(self, **args):
        """
        Generates a single model answer using greedy decoding.
        Parameters:
            args: input arguments for generation.
        Returns:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """ 
        args["temperature"] = 0.0
        args["n"] = 1

        output = self.generate_texts(**args)

        # For each input in batch model returns a list with one generation
        # we take the first generation from each list
        output = [out[0] for out in output]

        return output

    def __call__(self, **args):
        """
        Not implemented for blackbox models.
        """
        raise Exception("Cannot call blackbox model directly, use generate or generate_texts methods.")

    def tokenizer(self, *args, **kwargs):
        """
        Not implemented for blackbox models.
        """
        raise Exception("Cannot access tokenizer of blackbox model")
