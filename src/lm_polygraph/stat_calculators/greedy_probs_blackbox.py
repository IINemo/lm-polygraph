import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.blackbox_model import BlackboxModel


class BlackboxGreedyTextsCalculator(StatCalculator):
    """
    A calculator that handles both blackbox and greybox model generation.
    If the model supports logprobs, it provides full greybox functionality (texts, logprobs, etc.).
    Otherwise, it falls back to basic blackbox functionality (texts only).
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "greedy_texts",
            "greedy_log_probs",
            "greedy_log_likelihoods",
            "greedy_tokens",
            "greedy_tokens_alternatives",
        ], []

    def __init__(
        self,
        top_logprobs: int = 5,
    ):
        super().__init__()
        self.top_logprobs = top_logprobs

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates generation texts and log probabilities for the model.
        Provides full greybox functionality if the model supports logprobs,
        otherwise falls back to basic blackbox functionality.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (BlackboxModel): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with model generations and, if supported, probability data:
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_log_probs' (List[List[np.array]]): logits for the top k tokens (greybox only),
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of generated tokens (greybox only),
                - 'greedy_tokens' (List[List[str]]): tokens of the generated text (greybox only),
                - 'greedy_tokens_alternatives' (List[List[List[Tuple[str, float]]]]): alternative tokens with logprobs.
        """
        if model.supports_logprobs:
            # Greybox path: generate with logprobs
            output = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
                output_scores=True,
                top_logprobs=self.top_logprobs,
                temperature=0.0, # Greedy generation
            )

            # For each input in batch model returns a list with one generation
            # we take the first generation from each list
            output = [out[0] for out in output]

            # Process the results
            greedy_texts = [out.text for out in output]
            greedy_tokens = [out.tokens for out in output]
            greedy_log_likelihoods = [out.logprobs for out in output]
            greedy_log_probs = [out.top_logprobs for out in output]
            greedy_tokens_alternatives = []

            # iterate over batch
            for out in output:
                sample_alternatives = []
                # iterate over generation steps
                for i, step_alternatives in enumerate(out.alternative_tokens):
                    step_alternatives.sort(
                        key=lambda x: x == out.tokens[i],
                        reverse=True,
                    )
                    sample_alternatives.append(step_alternatives)
                greedy_tokens_alternatives.append(sample_alternatives)

            result = {
                "greedy_texts": greedy_texts,
                "greedy_log_probs": greedy_log_probs,
                "greedy_log_likelihoods": greedy_log_likelihoods,
                "greedy_tokens": greedy_tokens,
                "greedy_tokens_alternatives": greedy_tokens_alternatives,
            }
        else:
            # Blackbox path: generate just the text
            output = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
                temperature=0.0, # Greedy generation
            )

            # For each input in batch model returns a list with one generation
            # we take the first generation from each list
            output = [out[0] for out in output]

            # For blackbox models, only return the generated texts
            result = {"greedy_texts": [out.text for out in output]}

            # If user explicitly requests logprobs functionality but model doesn't support it, raise error
            if any(
                key in dependencies
                for key in [
                    "greedy_log_probs",
                    "greedy_log_likelihoods",
                    "greedy_tokens",
                    "greedy_tokens_alternatives",
                ]
            ):
                raise ValueError(
                    "Model must support logprobs for retrieving probability information. "
                    "The current model only supports text generation."
                )

        return result
