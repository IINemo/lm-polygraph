import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import BlackboxModel


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
            sequences = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
                output_scores=True,
                top_logprobs=self.top_logprobs,
            )
        else:
            # Blackbox path: generate just the text
            sequences = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
            )

        # Process the results
        greedy_texts = sequences
        greedy_log_probs = []
        greedy_log_likelihoods = []
        greedy_tokens = []
        greedy_tokens_alternatives = []

        # Extract logprobs and tokens from the model's stored data if available
        if model.supports_logprobs and hasattr(model, "logprobs") and model.logprobs:
            for i, logprob_data in enumerate(model.logprobs):
                if hasattr(logprob_data, "content"):
                    # Extract tokens
                    tokens = [item.token for item in logprob_data.content]
                    greedy_tokens.append(tokens)

                    # Extract log probabilities for this generation
                    log_probs_list = []
                    log_likelihoods = []

                    # Initialize alternatives structure for this generation
                    generation_alternatives = []

                    for token_logprobs in logprob_data.content:
                        # Get the top logprobs for this token position from OpenAI API
                        token_logprob_dict = {}

                        # Build alternatives for this token position
                        position_alternatives = []

                        # First add the chosen token with its logprob
                        chosen_token = token_logprobs.token
                        chosen_logprob = token_logprobs.logprob
                        position_alternatives.append((chosen_token, chosen_logprob))

                        # Then add the remaining top alternatives
                        for top_logprob in getattr(token_logprobs, "top_logprobs", []):
                            token_logprob_dict[top_logprob.token] = top_logprob.logprob
                            # Only add alternatives that are different from the chosen token
                            if top_logprob.token != chosen_token:
                                position_alternatives.append(
                                    (top_logprob.token, top_logprob.logprob)
                                )

                        # Add alternatives for this position to the generation alternatives
                        generation_alternatives.append(position_alternatives)

                        # Create a sparse representation of the logprobs distribution
                        sparse_logprobs = np.ones(50000) * -float("inf")

                        # Map token strings to positions in the sparse array
                        for token_str, logprob in token_logprob_dict.items():
                            token_idx = hash(token_str) % len(sparse_logprobs)
                            sparse_logprobs[token_idx] = logprob

                        log_probs_list.append(sparse_logprobs)
                        log_likelihoods.append(chosen_logprob)

                    greedy_log_probs.append(log_probs_list)
                    greedy_log_likelihoods.append(log_likelihoods)
                    greedy_tokens_alternatives.append(generation_alternatives)

            # Ensure all outputs have the same length for greybox case
            while len(greedy_tokens) < len(greedy_texts):
                # If we're missing token data, add placeholder empty lists
                greedy_tokens.append([])
                greedy_log_probs.append([])
                greedy_log_likelihoods.append([])
                greedy_tokens_alternatives.append([])

            result = {
                "greedy_texts": greedy_texts,
                "greedy_log_probs": greedy_log_probs,
                "greedy_log_likelihoods": greedy_log_likelihoods,
                "greedy_tokens": greedy_tokens,
                "greedy_tokens_alternatives": greedy_tokens_alternatives,
            }
        else:
            # For blackbox models, only return the generated texts
            result = {"greedy_texts": greedy_texts}

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
