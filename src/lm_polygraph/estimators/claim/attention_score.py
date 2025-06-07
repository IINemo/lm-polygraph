import numpy as np

from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.attention_score import unpad_attentions


class AttentionScoreClaim(Estimator):
    def __init__(self, layer: int = 16):
        super().__init__(
            ["forwardpass_attention_weights", "greedy_tokens", "claims"], "claim"
        )
        self.layer = layer

    def __str__(self):
        return f"AttentionScoreClaim (layer={self.layer})"

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
                ue[-1].append(-ue_i)
        return ue
