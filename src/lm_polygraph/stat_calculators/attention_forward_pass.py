import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class AttentionForwardPassCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * attention masks across the model (if applicable)
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return ["forwardpass_attention_weights"], ["greedy_tokens"]

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
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        cut_sequences = dependencies["greedy_tokens"]

        forwardpass_attention_weights = []

        for i in range(len(texts)):
            input_ids = torch.cat(
                [
                    batch["input_ids"][i].unsqueeze(0),
                    torch.tensor([cut_sequences[i]]).to(model.device()),
                ],
                axis=1,
            )
            with torch.no_grad():
                forwardpass_attentions = model.model(
                    input_ids, output_attentions=True
                ).attentions
                forwardpass_attentions = tuple(
                    attention.to("cpu") for attention in forwardpass_attentions
                )
                forwardpass_attentions = (
                    torch.cat(forwardpass_attentions).float().numpy()
                )
            forwardpass_attention_weights.append(forwardpass_attentions)
        try:
            forwardpass_attention_weights = np.array(forwardpass_attention_weights)
        except Exception:
            # in this case we have various len of input_ids+greedy_tokens in batch, so pad before concat
            max_seq_length = np.max(
                [el.shape[-1] for el in forwardpass_attention_weights]
            )
            forwardpass_attention_weights_padded = []
            for el in forwardpass_attention_weights:
                buf_el = el
                if el.shape[-1] != max_seq_length:
                    pad_mask = (
                        (0, 0),
                        (0, 0),
                        (0, max_seq_length - el.shape[-1]),
                        (0, max_seq_length - el.shape[-1]),
                    )
                    buf_el = np.pad(el, pad_mask, constant_values=np.nan)
                forwardpass_attention_weights_padded.append(buf_el)
            forwardpass_attention_weights = np.array(
                forwardpass_attention_weights_padded
            )

        result_dict = {"forwardpass_attention_weights": forwardpass_attention_weights}
        return result_dict
