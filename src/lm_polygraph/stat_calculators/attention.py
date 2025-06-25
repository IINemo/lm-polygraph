import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class BaseAttentionCalculator(StatCalculator):
    """
    Calculates and aggregates attention maps for a batch of input texts using a WhiteboxModel.

    This calculator extracts the raw attention weights from the model and processes them to produce
    a summary attention map for each input sequence. The output is a list of attention matrices,
    one per input in the batch, where each matrix summarizes the attention across all heads
    and layers for each token position.

    Returns:
        - "attention_all": List[np.ndarray]
            Each element is a (num_heads * num_layers, seq_len, seq_len) array representing the
            stacked attention maps for a single input sequence.

    Dependencies:
        - "attention_raw": Raw attention weights from the model.
        - "greedy_tokens": List of tokenized input sequences.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.

        Returns:
            Tuple[List[str], List[str]]: (list of statistics produced, list of dependencies required)
        """
        return [
            "attention_all",
        ], ["attention_raw"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Computes aggregated attention maps for each input sequence in the batch.

        Args:
            dependencies (Dict[str, np.ndarray]): Dictionary containing required dependencies,
                including "attention_raw" (raw attention weights) and "greedy_tokens" (tokenized sequences).
            texts (List[str]): Batch of input texts.
            model (WhiteboxModel): The model used to generate attention weights.
            max_new_tokens (int, optional): Maximum number of new tokens to generate (unused). Default is 100.

        Returns:
            Dict[str, List[np.ndarray]]: Dictionary with key "attention_all" mapping to a list of
                attention matrices (one per input sequence), each of shape
                (num_heads * num_layers, seq_len, seq_len).
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        attentions = dependencies["attention_raw"]
        cut_sequences = dependencies["greedy_tokens"]

        attention_all = []
        if model.model_type != "vLLMCausalLM":
            for i in range(len(cut_sequences)):
                c = len(cut_sequences[i])
                attn_mask = np.zeros(
                    shape=(
                        model.model.config.num_attention_heads
                        * model.model.config.num_hidden_layers,
                        c,
                        c,
                    )
                )
                for j in range(1, c):
                    stacked_attention = torch.vstack(
                        [
                            attentions[j][layer][0][head][0][-j:]
                            for layer in range(len(attentions[j]))
                            for head in range(len(attentions[j][layer][0]))
                        ]
                    )
                    if stacked_attention.dtype == torch.bfloat16:
                        stacked_attention = stacked_attention.to(
                            torch.float16
                        )  # numpy does not support bfloat16
                    attn_mask[:, j, :j] = stacked_attention.cpu().numpy()
                attention_all.append(attn_mask.max(0))
        result_dict = {
            "attention_all": attention_all,
        }
        return result_dict


class LookbackRatioCalculator(StatCalculator):
    """
    Computes the lookback ratio for each token position in a batch of input texts using a WhiteboxModel.

    The lookback ratio quantifies the proportion of attention paid to the context (previous tokens)
    versus the newly generated token at each step, for each head and layer. This can be used to
    analyze how much the model "looks back" at previous context when generating new tokens.

    Returns:
        - "lookback_ratios": List[List[float]]
            Each element is a list of lookback ratios for a single token position, across all heads and layers.

    Dependencies:
        - "attention_raw": Raw attention weights from the model.
        - "greedy_tokens": List of tokenized input sequences.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.

        Returns:
            Tuple[List[str], List[str]]: (list of statistics produced, list of dependencies required)
        """
        return [
            "lookback_ratios",
        ], ["attention_raw"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Computes the lookback ratio for each token position in the batch.

        For each token position, for each head and layer, the lookback ratio is defined as:
            lookback_ratio = attention_on_context / (attention_on_new + attention_on_context)
        where:
            - attention_on_context: mean attention paid to previous tokens (context)
            - attention_on_new: mean attention paid to the new token

        Args:
            dependencies (Dict[str, np.ndarray]): Dictionary containing required dependencies,
                including "attention_raw" (raw attention weights) and "greedy_tokens" (tokenized sequences).
            texts (List[str]): Batch of input texts.
            model (WhiteboxModel): The model used to generate attention weights.
            max_new_tokens (int, optional): Maximum number of new tokens to generate (unused). Default is 100.

        Returns:
            Dict[str, List[List[float]]]: Dictionary with key "lookback_ratios" mapping to a list of
                lists, where each inner list contains lookback ratios for a token position across all heads and layers.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        attentions = dependencies["attention_raw"]
        cut_sequences = dependencies["greedy_tokens"]

        lookback_ratios = []
        if model.model_type != "vLLMCausalLM":
            for i in range(len(cut_sequences)):
                c = len(cut_sequences[i])
                for j in range(c):
                    lookback_ratios_token = []
                    for layer in range(len(attentions[j])):
                        for head in range(len(attentions[j][layer][0])):
                            if j:
                                attention_on_new = (
                                    attentions[j][layer][0][head][0][-j:].mean().item()
                                )
                                attention_on_context = (
                                    attentions[j][layer][0][head][0][:-j].mean().item()
                                )
                            else:
                                attention_on_new = 0
                                attention_on_context = (
                                    attentions[j][layer][0][head][0].mean().item()
                                )
                            lookback_ratio = attention_on_context / (
                                attention_on_new + attention_on_context
                            )
                            lookback_ratios_token.append(lookback_ratio)
                    lookback_ratios.append(lookback_ratios_token)
        result_dict = {
            "lookback_ratios": lookback_ratios,
        }
        return result_dict
