import torch
import numpy as np

from typing import Dict, List

from stat_calculators.stat_calculator import StatCalculator
from utils.model import Model


class GreedyLMProbsCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['greedy_lm_log_probs', 'greedy_lm_log_likelihoods'], ['greedy_tokens'])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
        tokens = dependencies['greedy_tokens']
        batch = model.tokenize([model.tokenizer.decode(t) for t in tokens])
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            logprobs = model.model(**batch).logits.log_softmax(-1)
        greedy_lm_log_probs = []
        greedy_lm_ll = []
        for i in range(len(tokens)):
            assert len(logprobs[i]) >= len(tokens[i])
            greedy_lm_log_probs.append(logprobs[i, -len(tokens[i]):-1].cpu().numpy())
            greedy_lm_ll.append([logprobs[i, -len(tokens[i]) + j, tokens[i][j]].item()
                                 for j in range(len(tokens[i]))])
        return {
            'greedy_lm_log_probs': greedy_lm_log_probs,
            'greedy_lm_log_likelihoods': greedy_lm_ll,
        }
