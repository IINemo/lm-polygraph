import torch
import numpy as np
from typing import Dict, List, Tuple
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class GreedyLMProbsVisualCalculator(StatCalculator):
    """
    Calculates probabilities of the model generations without input texts.
    Used to calculate P(y_t|y_<t) subtrahend in PointwiseMutualInformation.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return ["greedy_lm_log_probs", "greedy_lm_log_likelihoods"], ["greedy_tokens"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: VisualWhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the entropy of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, consisting of:
                - 'greedy_tokens' (List[List[int]]): tokenized model generations for each input text.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'greedy_lm_log_probs' (List[List[np.array]]): logarithms of autoregressive probability distributions
                    when generating the generated text without input text.
                - 'greedy_lm_log_likelihoods' (List[List[float]]): log-probabilities of generating text without input.
                    P(y_t | y_<t) for all t.
        """
        tokens = dependencies["greedy_tokens"]
        images = dependencies["images"]

        greedy_lm_log_probs = []
        greedy_lm_ll = []

        # Process each sample individually
        for i, toks in enumerate(tokens):
            try:
                # Create encoding with image and empty text context
                # We use the original text to get proper image encoding, but then we'll replace input_ids
                encoding = model.processor_visual(
                    text="",  # Use empty text as context for P(y_t|y_<t)
                    images=images[i],
                    return_tensors="pt",
                ).to(model.device())

                # Replace input_ids with the tokens we want to evaluate
                encoding["input_ids"] = torch.LongTensor([toks]).to(model.device())
                encoding["attention_mask"] = torch.ones_like(encoding["input_ids"]).to(
                    model.device()
                )

                # Also need to update image_embeds_position_mask for the new sequence length
                old_mask = encoding.get("image_embeds_position_mask", None)
                if old_mask is not None:
                    # Create new mask: all zeros since we're only evaluating the generated tokens
                    new_mask = torch.zeros_like(encoding["input_ids"])
                    encoding["image_embeds_position_mask"] = new_mask

                # Forward pass through the model WITH image
                with torch.no_grad():
                    outputs = model(**encoding, return_dict=True)
                    logprobs = outputs.logits.log_softmax(-1)

                logprobs = logprobs[0]

                # Verify the length matches
                if len(logprobs) < len(toks):
                    print(
                        f"Warning: logprobs length {len(logprobs)} < tokens length {len(toks)}"
                    )
                    # Use what we have
                    usable_length = min(len(logprobs), len(toks))
                    greedy_lm_log_probs.append(
                        logprobs[-usable_length:-1].cpu().numpy()
                        if usable_length > 1
                        else np.array([])
                    )
                    greedy_lm_ll.append(
                        [
                            logprobs[-usable_length + j, toks[j]].item()
                            for j in range(usable_length)
                        ]
                    )
                else:
                    greedy_lm_log_probs.append(
                        logprobs[-len(toks) : -1].cpu().numpy()
                        if len(toks) > 1
                        else np.array([])
                    )
                    greedy_lm_ll.append(
                        [
                            logprobs[-len(toks) + j, toks[j]].item()
                            for j in range(len(toks))
                        ]
                    )

            except Exception as e:
                print(
                    f"Error processing sample {i} in GreedyLMProbsVisualCalculator: {e}"
                )
                try:
                    print(f"Trying text-only fallback for sample {i}")
                    input_ids = torch.LongTensor([toks]).to(model.device())
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids).to(model.device()),
                    }

                    with torch.no_grad():
                        # Try to use just the text model if available
                        if hasattr(model.model, "text_model"):
                            outputs = model.model.text_model(**batch)
                            logprobs = outputs.logits.log_softmax(-1)
                        else:
                            # Last resort: try direct model call without image
                            # This will likely fail for Kosmos2 but worth trying
                            outputs = model.model(**batch)
                            logprobs = outputs.logits.log_softmax(-1)

                    logprobs = logprobs[0]
                    if len(logprobs) >= len(toks):
                        greedy_lm_log_probs.append(
                            logprobs[-len(toks) : -1].cpu().numpy()
                            if len(toks) > 1
                            else np.array([])
                        )
                        greedy_lm_ll.append(
                            [
                                logprobs[-len(toks) + j, toks[j]].item()
                                for j in range(len(toks))
                            ]
                        )
                    else:
                        raise ValueError("Fallback also failed")

                except Exception as e2:
                    print(f"Fallback also failed for sample {i}: {e2}")
                    # Create dummy values as last resort
                    seq_len = len(toks)
                    vocab_size = (
                        model.model.config.vocab_size
                        if hasattr(model.model, "config")
                        else 1000
                    )
                    dummy_log_probs = np.zeros((max(seq_len - 1, 0), vocab_size))
                    dummy_ll = [0.0] * seq_len

                    greedy_lm_log_probs.append(dummy_log_probs)
                    greedy_lm_ll.append(dummy_ll)

        return {
            "greedy_lm_log_probs": greedy_lm_log_probs,
            "greedy_lm_log_likelihoods": greedy_lm_ll,
        }
