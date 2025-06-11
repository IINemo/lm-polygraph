import numpy as np

from typing import Dict
from transformers import AutoConfig

from lm_polygraph.estimators.estimator import Estimator


def unpad_attentions(forwardpass_attention_weights_original):
    """Unpad batched and padded with np.nan attentions."""
    forwardpass_attention_weights = []
    for el in forwardpass_attention_weights_original:
        buf_el = el
        if np.isnan(el).any():
            # unpad
            initial_shape = (
                el.shape[0],
                el.shape[1],
                (~np.isnan(el)[0][0][0]).sum(),
                (~np.isnan(el)[0][0][0]).sum(),
            )
            buf_el = el[~np.isnan(el)].reshape(initial_shape)
        forwardpass_attention_weights.append(buf_el)
    return forwardpass_attention_weights


class AttentionScore(Estimator):
    """
    Estimates uncertainty based on model's attention weights as in
    Attention Score method from https://openreview.net/forum?id=LYx4w3CAgy
    """

    def __init__(
        self,
        model_name: str = None,
        gen_only: bool = False,
    ):
        super().__init__(["forwardpass_attention_weights", "greedy_tokens"], "sequence")
        if model_name is not None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.layer = config.num_hidden_layers // 2  # middle layer
        else:
            raise ValueError("model_name must be provided to initialize self.layer")
        self.gen_only = gen_only

    def __str__(self):
        if self.gen_only:
            return f"AttentionScore gen-only (layer={self.layer})"
        return f"AttentionScore (layer={self.layer})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        forwardpass_attention_weights_original = stats["forwardpass_attention_weights"]
        # check nan and unpad
        forwardpass_attention_weights = unpad_attentions(
            forwardpass_attention_weights_original
        )
        greedy_tokens = stats["greedy_tokens"]
        ue = []
        for k, attention_weight in enumerate(forwardpass_attention_weights):
            ue_i = 0
            for attn in attention_weight[self.layer]:
                if self.gen_only:
                    attn = attn[
                        -len(greedy_tokens[k]) : -1, -len(greedy_tokens[k]) : -1
                    ]  # USE ONLY GENERATED TOKENS

                ue_i += np.sum(np.log(np.diag(attn)))
            ue_i /= len(attention_weight[self.layer])
            ue.append(ue_i)
        return -np.array(ue)
