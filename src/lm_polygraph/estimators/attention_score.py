import numpy as np
from typing import Dict
from lm_polygraph.estimators.estimator import Estimator


def unpad_attentions(forwardpass_attention_weights_original):
    """Unpad batched and padded with np.nan attentions."""
    forwardpass_attention_weights = []
    for el in forwardpass_attention_weights_original:
        buf_el = el
        if np.isnan(el).any():
            # Handle different possible shapes
            if el.ndim == 4:
                # Shape: (layers, heads, seq_len, seq_len)
                initial_shape = (
                    el.shape[0],
                    el.shape[1],
                    (~np.isnan(el)[0][0][0]).sum(),
                    (~np.isnan(el)[0][0][0]).sum(),
                )
                buf_el = el[~np.isnan(el)].reshape(initial_shape)
            elif el.ndim == 5:
                # Shape from visual model: (layers, batch=1, heads, seq_len, seq_len)
                # Squeeze the batch dimension first
                el_squeezed = el[:, 0, :, :, :]  # Remove batch dimension
                initial_shape = (
                    el_squeezed.shape[0],
                    el_squeezed.shape[1],
                    (~np.isnan(el_squeezed)[0][0][0]).sum(),
                    (~np.isnan(el_squeezed)[0][0][0]).sum(),
                )
                buf_el = el_squeezed[~np.isnan(el_squeezed)].reshape(initial_shape)
            else:
                print(f"Warning: Unexpected attention shape {el.shape}, skipping unpad")
        forwardpass_attention_weights.append(buf_el)
    return forwardpass_attention_weights


class AttentionScore(Estimator):
    """
    Estimates uncertainty based on model's attention weights as in
    Attention Score method from https://openreview.net/forum?id=LYx4w3CAgy
    """

    def __init__(
        self,
        layer: int = None,
        gen_only: bool = False,
    ):
        super().__init__(["forwardpass_attention_weights", "greedy_tokens"], "sequence")
        self.layer = layer
        self.gen_only = gen_only

    def __str__(self):
        if self.gen_only:
            return f"AttentionScore gen-only (layer={self.layer})"
        return f"AttentionScore (layer={self.layer})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if self.layer is None:
            self.layer = stats["model"].model.config.num_hidden_layers // 2

        forwardpass_attention_weights_original = stats["forwardpass_attention_weights"]
        # check nan and unpad
        forwardpass_attention_weights = unpad_attentions(
            forwardpass_attention_weights_original
        )
        greedy_tokens = stats["greedy_tokens"]
        ue = []

        for k, attention_weight in enumerate(forwardpass_attention_weights):
            ue_i = 0
            # Handle different attention weight shapes
            if attention_weight.ndim == 4:
                # Standard shape: (layers, heads, seq_len, seq_len)
                layer_attention = attention_weight[self.layer]
                num_heads = layer_attention.shape[0]

                for head_idx in range(num_heads):
                    attn = layer_attention[head_idx]
                    if attn.ndim != 2:
                        print(
                            f"Warning: Skipping non-2D attention matrix with shape {attn.shape}"
                        )
                        continue

                    if self.gen_only:
                        attn = attn[
                            -len(greedy_tokens[k]) : -1, -len(greedy_tokens[k]) : -1
                        ]  # USE ONLY GENERATED TOKENS

                    # Ensure we have a valid 2D matrix before taking diagonal
                    if attn.ndim == 2 and attn.shape[0] == attn.shape[1]:
                        diag_vals = np.diag(attn)
                        # Add small epsilon to avoid log(0)
                        ue_i += np.sum(np.log(diag_vals + 1e-12))
                    else:
                        print(f"Warning: Invalid attention matrix shape {attn.shape}")

                ue_i /= num_heads

            elif attention_weight.ndim == 5:
                # Visual model shape: (layers, batch=1, heads, seq_len, seq_len)
                # Take the first (and only) batch element
                layer_attention = attention_weight[self.layer, 0, :, :, :]
                num_heads = layer_attention.shape[0]

                for head_idx in range(num_heads):
                    attn = layer_attention[head_idx]
                    if attn.ndim != 2:
                        print(
                            f"Warning: Skipping non-2D attention matrix with shape {attn.shape}"
                        )
                        continue

                    if self.gen_only:
                        attn = attn[
                            -len(greedy_tokens[k]) : -1, -len(greedy_tokens[k]) : -1
                        ]

                    if attn.ndim == 2 and attn.shape[0] == attn.shape[1]:
                        diag_vals = np.diag(attn)
                        ue_i += np.sum(np.log(diag_vals + 1e-12))
                    else:
                        print(f"Warning: Invalid attention matrix shape {attn.shape}")

                ue_i /= num_heads
            else:
                print(
                    f"Warning: Unexpected attention weight shape {attention_weight.shape}"
                )
                ue_i = 0  # Default value for invalid shapes

            ue.append(ue_i)
        return -np.array(ue)
