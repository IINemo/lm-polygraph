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
        requires_logprobs = any(
            args.get(arg, False)
            for arg in [
                "output_scores",
                "output_attentions",
                "output_hidden_states",
            ]
        )

        # Use adapter to check logprobs support (considering model-specific rules)
        if requires_logprobs and not self.supports_logprobs:
            raise Exception(
                f"Cannot access logits for blackbox model with provider '{self.api_provider_name}'"
            )

        return self.adapter.generate_texts(self, input_texts, args)

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
