import numpy as np

from typing import Dict
from transformers import AutoConfig

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators.attention_score import unpad_attentions


class AttentionScoreClaim(Estimator):
    def __init__(
        self,
        model_name: str = None,
    ):
        super().__init__(
            ["forwardpass_attention_weights", "greedy_tokens", "claims"], "claim"
        )
        if model_name is not None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.layer = config.num_hidden_layers // 2  # middle layer
        else:
            raise ValueError("model_name must be provided to initialize self.layer")

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
