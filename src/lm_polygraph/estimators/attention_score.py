import numpy as np

from typing import Dict

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
        layer: int = 16,
        gen_only: bool = False,
    ):
        super().__init__(["forwardpass_attention_weights", "greedy_tokens"], "sequence")
        self.layer = layer
        self.gen_only = gen_only

    def __str__(self):
        if self.gen_only:
            return f"LLMCheckAttentionGEN Layer {self.layer}"
        return f"LLMCheckAttention Layer {self.layer}"

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


class AttentionScoreClaim(Estimator):
    def __init__(self, layer: int = 16):
        super().__init__(["forwardpass_attention_weights", "greedy_tokens", "claims"], "claim")
        self.layer = layer

    def __str__(self):
        return f"LLMCheckAttentionClaim Layer {self.layer}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        forwardpass_attention_weights_original = stats["forwardpass_attention_weights"]
        # check nan and unpad
        forwardpass_attention_weights = unpad_attentions(
            forwardpass_attention_weights_original
        )
        greedy_tokens = stats["greedy_tokens"]
        claims = stats["claims"]

        ue = []
        for k, attention_weight in enumerate(forwardpass_attention_weights):
            ue.append([])
            for claim in claims[k]:
                ue_i = 0
                tokens = np.array(claim.aligned_token_ids)
                for attn in attention_weight[self.layer]:
                    attn = attn[-len(greedy_tokens[k]) :, -len(greedy_tokens[k]) :]
                    ue_i += np.sum(np.log(np.diag(attn)[tokens]))
                ue_i /= len(attention_weight[self.layer])
                ue[-1].append(ue_i)
        return -np.array(ue)
