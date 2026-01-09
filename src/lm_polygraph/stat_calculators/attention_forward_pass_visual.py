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
                outputs = model(**encoding)

                # Safely get attentions
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    # Convert to CPU and numpy
                    attentions_cpu = []
                    for attention in outputs.attentions:
                        if attention is not None:
                            attentions_cpu.append(attention.cpu().float())
                        else:
                            # Create dummy attention if None
                            batch_size = encoding["input_ids"].shape[0]
                            seq_len = encoding["input_ids"].shape[1]

                            # Get model config for proper dimensions
                            try:
                                num_heads = model.model.config.num_attention_heads
                            except Exception:  # Fixed: removed unused 'e'
                                num_heads = 12  # fallback

                            # Create identity matrix for dummy attention
                            dummy_attn = (
                                torch.eye(seq_len, device="cpu")
                                .unsqueeze(0)
                                .unsqueeze(0)
                            )
                            dummy_attn = dummy_attn.expand(
                                batch_size, num_heads, seq_len, seq_len
                            )
                            attentions_cpu.append(dummy_attn)

                    if attentions_cpu:
                        # Stack along layer dimension
                        forwardpass_attentions = torch.stack(
                            attentions_cpu, dim=0
                        ).numpy()
                    else:
                        # Fallback if no attentions
                        batch_size, seq_len = encoding["input_ids"].shape
                        try:
                            num_layers = model.model.config.num_hidden_layers
                            num_heads = model.model.config.num_attention_heads
                        except Exception:  # Fixed: removed unused 'e'
                            num_layers = 12
                            num_heads = 12
                        forwardpass_attentions = np.ones(
                            (num_layers, batch_size, num_heads, seq_len, seq_len)
                        )
                else:
                    # Fallback if model didn't return attentions
                    batch_size, seq_len = encoding["input_ids"].shape
                    try:
                        num_layers = model.model.config.num_hidden_layers
                        num_heads = model.model.config.num_attention_heads
                    except Exception:  # Fixed: removed unused 'e'
                        num_layers = 12
                        num_heads = 12
                    forwardpass_attentions = np.ones(
                        (num_layers, batch_size, num_heads, seq_len, seq_len)
                    )

                forwardpass_attention_weights.append(forwardpass_attentions)

        # Handle padding if sequence lengths vary
        try:
            forwardpass_attention_weights = np.array(forwardpass_attention_weights)
        except Exception:  # Fixed: removed unused 'e'
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
