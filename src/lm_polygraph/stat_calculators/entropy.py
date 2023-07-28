import torch
import numpy as np
import torch.nn.functional as F

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class EntropyCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['entropy'], ['greedy_log_probs'])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: WhiteboxModel, **kwargs) -> Dict[str, np.ndarray]:
        logprobs = dependencies['greedy_log_probs']
        entropies = []
        for s_lp in logprobs:
            entropies.append([])
            for lp in s_lp:
                mask = ~np.isinf(lp)
                entropies[-1].append(-np.mean(np.array(lp[mask]) * np.exp(lp[mask])))
        return {'entropy': entropies}
