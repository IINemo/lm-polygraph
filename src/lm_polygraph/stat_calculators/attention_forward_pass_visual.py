import torch
import numpy as np
from typing import Dict, List, Tuple
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel


class AttentionForwardPassCalculatorVisual(StatCalculator):
    """
    For VisualLM (vision-language) WhiteboxModel: computes attention weights
    during forward pass using greedy tokens + image inputs.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["forwardpass_attention_weights"], ["greedy_tokens"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: VisualWhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Parameters:
            dependencies: includes greedy_tokens and images.
            texts: List of input texts
            model: VisualWhiteboxModel wrapper.
        Returns:
            Dict with 'forwardpass_attention_weights' numpy array.
        """
        images = dependencies["images"]
        forwardpass_attention_weights = []

        for i in range(len(texts)):
            # Process each sample individually
            encoding = model.processor_visual(
                text=str(texts[i]), images=images[i], return_tensors="pt"
            ).to(model.device())

            # Forward pass with attention output
            with torch.no_grad():
                forwardpass_attentions = model.model(
                    **encoding, output_attentions=True, output_hidden_states=True
                ).attentions
                forwardpass_attentions = tuple(
                    attention.to("cpu") for attention in forwardpass_attentions
                )
                forwardpass_attentions = (
                    torch.cat(forwardpass_attentions).float().numpy()
                )
            forwardpass_attention_weights.append(forwardpass_attentions)

        # Handle padding if sequence lengths vary
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
        return {"forwardpass_attention_weights": forwardpass_attention_weights}
