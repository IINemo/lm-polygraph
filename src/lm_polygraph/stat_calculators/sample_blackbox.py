from typing import Dict, List, Tuple

import numpy as np

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters import BlackboxModel


class BlackboxSamplingGenerationCalculator(StatCalculator):
    """
    Calculates several sampled texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "sample_log_probs",
            "sample_tokens",
            "sample_texts",
            "sample_log_likelihoods",
        ], []

    def __init__(self, samples_n: int = 10):
        super().__init__()
        self.samples_n = samples_n

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates sampled texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[float]]): sum of the log probabilities at each token of the sampling generation.
                - 'sample_log_likelihoods' (List[List[List[float]]]): log probabilities at each token of the sampling generation.
        """
        if model.supports_logprobs:
            # Greybox path: generate with logprobs
            output = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=self.samples_n,
                output_scores=True,
            )

            sample_texts = []
            sample_log_probs = []
            sample_log_likelihoods = []
            sample_tokens = []

            # Iterating over batch
            for out in output:
                sample_texts.append([sample_out.text for sample_out in out])
                sample_tokens.append([sample_out.tokens for sample_out in out])
                sample_log_probs.append(
                    [np.sum(sample_out.logprobs) for sample_out in out]
                )
                sample_log_likelihoods.append(
                    [sample_out.logprobs for sample_out in out]
                )

            result = {
                "sample_texts": sample_texts,
                "sample_log_probs": sample_log_probs,
                "sample_log_likelihoods": sample_log_likelihoods,
                "sample_tokens": sample_tokens,
            }
        else:
            # Blackbox path: generate just the text
            output = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=self.samples_n,
            )

            sample_texts = []
            for out in output:
                sample_texts.append([sample_out.text for sample_out in out])

            result = {
                "sample_texts": sample_texts,
            }

            # If user explicitly requests logprobs functionality but model doesn't support it, raise error
            if any(
                key in dependencies
                for key in [
                    "sample_log_probs",
                    "sample_log_likelihoods",
                    "sample_tokens",
                ]
            ):
                raise ValueError(
                    "Model must support logprobs for retrieving probability information. "
                    "The current model only supports text generation."
                )

        return result
